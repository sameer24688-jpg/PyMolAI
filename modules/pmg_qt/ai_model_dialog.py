from __future__ import annotations

from pymol.Qt import QtCore, QtWidgets

from pymol.ai.model_catalog import (
    ModelCatalogError,
    ModelEntry,
    fetch_provider_catalog,
    filter_models,
    merge_favorites_and_catalog,
)
from pymol.ai.model_store import (
    delete_saved_model,
    load_saved_models,
    models_for_menu,
    upsert_saved_model,
)
from pymol.ai.provider_key_store import resolve_api_key
from pymol.ai.providers import active_provider_id, get_provider_spec, provider_menu_entries


class _CatalogWorker(QtCore.QObject):
    finished = QtCore.Signal(bool, str, list)

    def __init__(self, provider_id: str, api_key: str):
        super().__init__()
        self._provider_id = provider_id
        self._api_key = api_key

    @QtCore.Slot()
    def run(self):
        try:
            rows = fetch_provider_catalog(self._provider_id, api_key=self._api_key)
            self.finished.emit(True, "", rows)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc), [])


class AiModelDialog(QtWidgets.QDialog):
    """ONLYOFFICE-inspired Edit AI models dialog."""

    def __init__(self, parent=None, *, runtime=None, on_changed=None):
        super().__init__(parent)
        self._runtime = runtime
        self._on_changed = on_changed
        self._catalog_rows: list = []
        self._worker = None
        self._worker_thread = None

        self.setWindowTitle("Edit AI Models")
        self.setModal(True)
        self.resize(640, 420)

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form = QtWidgets.QFormLayout()
        self.provider_combo = QtWidgets.QComboBox(self)
        for pid, name in provider_menu_entries():
            self.provider_combo.addItem("%s (%s)" % (name, pid), pid)
        form.addRow("Provider", self.provider_combo)

        self.search_input = QtWidgets.QLineEdit(self)
        self.search_input.setPlaceholderText("Filter models…")
        form.addRow("Search", self.search_input)

        self.model_combo = QtWidgets.QComboBox(self)
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        form.addRow("Model", self.model_combo)

        self.display_name_input = QtWidgets.QLineEdit(self)
        self.display_name_input.setPlaceholderText("Optional friendly name")
        form.addRow("Display name", self.display_name_input)

        self.custom_id_input = QtWidgets.QLineEdit(self)
        self.custom_id_input.setPlaceholderText("Or type a custom model id")
        form.addRow("Custom model id", self.custom_id_input)
        layout.addLayout(form)

        self.saved_list = QtWidgets.QListWidget(self)
        self.saved_list.setMinimumHeight(120)
        layout.addWidget(QtWidgets.QLabel("Saved models for provider", self))
        layout.addWidget(self.saved_list)

        buttons = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Update models list", self)
        self.add_button = QtWidgets.QPushButton("Add / Update", self)
        self.delete_button = QtWidgets.QPushButton("Delete selected", self)
        self.use_button = QtWidgets.QPushButton("Use as active model", self)
        self.close_button = QtWidgets.QPushButton("Close", self)
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)
        buttons.addWidget(self.use_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        current_provider = getattr(runtime, "provider", None) if runtime is not None else None
        current_provider = current_provider or active_provider_id()
        idx = self.provider_combo.findData(current_provider)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)

        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.search_input.textChanged.connect(self._repopulate_model_combo)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        self.saved_list.itemSelectionChanged.connect(self._on_saved_selected)
        self.refresh_button.clicked.connect(self._on_refresh)
        self.add_button.clicked.connect(self._on_add)
        self.delete_button.clicked.connect(self._on_delete)
        self.use_button.clicked.connect(self._on_use)
        self.close_button.clicked.connect(self.accept)

        self._on_provider_changed()

    def _provider_id(self) -> str:
        return str(self.provider_combo.currentData() or "openrouter")

    def _notify(self):
        if callable(self._on_changed):
            self._on_changed()

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _on_provider_changed(self):
        self._catalog_rows = []
        self._refresh_saved_list()
        self._repopulate_model_combo()
        spec = get_provider_spec(self._provider_id())
        self._set_status(
            "Provider: %s | API style: %s | Favorites + saved models shown. "
            "Use Update models list to fetch remote catalog."
            % (spec.name, spec.api_style)
        )

    def _refresh_saved_list(self):
        self.saved_list.clear()
        for item in load_saved_models(self._provider_id()):
            label = "%s (%s)" % (item.display_name, item.model_id)
            row = QtWidgets.QListWidgetItem(label)
            row.setData(QtCore.Qt.UserRole, item.model_id)
            self.saved_list.addItem(row)

    def _available_entries(self) -> list:
        pid = self._provider_id()
        return merge_favorites_and_catalog(pid, self._catalog_rows)

    def _repopulate_model_combo(self):
        query = str(self.search_input.text() or "")
        entries = filter_models(self._available_entries(), query)
        current = str(self.model_combo.currentText() or "").strip()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for entry in entries:
            label = "%s — %s" % (entry.display_name, entry.model_id)
            self.model_combo.addItem(label, entry.model_id)
        self.model_combo.blockSignals(False)
        if current:
            idx = self.model_combo.findData(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            else:
                self.model_combo.setEditText(current)

    def _on_model_combo_changed(self):
        mid = str(self.model_combo.currentData() or self.model_combo.currentText() or "").strip()
        if not mid:
            return
        # If label form "name — id", prefer data role.
        data = self.model_combo.currentData()
        if data:
            self.custom_id_input.setText(str(data))
            # Try to fill display name from entry
            for entry in self._available_entries():
                if entry.model_id == str(data):
                    self.display_name_input.setText(entry.display_name)
                    break

    def _on_saved_selected(self):
        items = self.saved_list.selectedItems()
        if not items:
            return
        mid = str(items[0].data(QtCore.Qt.UserRole) or "").strip()
        if not mid:
            return
        self.custom_id_input.setText(mid)
        for saved in load_saved_models(self._provider_id()):
            if saved.model_id == mid:
                self.display_name_input.setText(saved.display_name)
                break

    def _selected_model_id(self) -> str:
        custom = str(self.custom_id_input.text() or "").strip()
        if custom:
            return custom
        data = self.model_combo.currentData()
        if data:
            return str(data).strip()
        text = str(self.model_combo.currentText() or "").strip()
        if " — " in text:
            return text.split(" — ", 1)[-1].strip()
        return text

    def _on_refresh(self):
        if self._worker_thread is not None:
            return
        pid = self._provider_id()
        key = resolve_api_key(pid)
        self.refresh_button.setEnabled(False)
        self._set_status("Refreshing model catalog for %s…" % (pid,))
        worker = _CatalogWorker(pid, key)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_catalog_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._worker = worker
        self._worker_thread = thread
        thread.start()

    @QtCore.Slot(bool, str, list)
    def _on_catalog_finished(self, ok: bool, message: str, rows: list):
        self._worker = None
        self._worker_thread = None
        self.refresh_button.setEnabled(True)
        if not ok:
            # Fail closed: keep previous catalog/favorites/saved models.
            self._set_status("Catalog refresh failed (local models unchanged): %s" % (message,))
            QtWidgets.QMessageBox.warning(self, "Update models list", str(message or "Refresh failed."))
            return
        self._catalog_rows = list(rows or [])
        self._repopulate_model_combo()
        self._set_status("Loaded %d remote models. Favorites remain listed first." % (len(self._catalog_rows),))

    def _on_add(self):
        mid = self._selected_model_id()
        if not mid:
            QtWidgets.QMessageBox.warning(self, "Add model", "Choose or enter a model id.")
            return
        name = str(self.display_name_input.text() or "").strip() or mid
        try:
            upsert_saved_model(provider_id=self._provider_id(), model_id=mid, display_name=name)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Add model", str(exc))
            return
        self._refresh_saved_list()
        self._notify()
        self._set_status("Saved model %s." % (mid,))

    def _on_delete(self):
        items = self.saved_list.selectedItems()
        mid = ""
        if items:
            mid = str(items[0].data(QtCore.Qt.UserRole) or "").strip()
        if not mid:
            mid = self._selected_model_id()
        if not mid:
            QtWidgets.QMessageBox.warning(self, "Delete model", "Select a saved model to delete.")
            return
        if not delete_saved_model(provider_id=self._provider_id(), model_id=mid):
            QtWidgets.QMessageBox.information(self, "Delete model", "Model was not in the saved list.")
            return
        self._refresh_saved_list()
        self._notify()
        self._set_status("Deleted saved model %s." % (mid,))

    def _on_use(self):
        mid = self._selected_model_id()
        if not mid:
            QtWidgets.QMessageBox.warning(self, "Use model", "Choose or enter a model id.")
            return
        name = str(self.display_name_input.text() or "").strip() or mid
        try:
            upsert_saved_model(provider_id=self._provider_id(), model_id=mid, display_name=name)
        except Exception:
            pass
        runtime = self._runtime
        if runtime is not None and hasattr(runtime, "set_model"):
            # Ensure provider matches selected provider when applying model.
            if hasattr(runtime, "set_provider") and runtime.provider != self._provider_id():
                runtime.set_provider(self._provider_id(), emit_notice=True, reset_model=False)
            runtime.set_model(mid, emit_notice=True)
        self._refresh_saved_list()
        self._notify()
        self._set_status("Active model set to %s." % (mid,))
        QtWidgets.QMessageBox.information(self, "Use model", "Active chat model set to:\n%s" % (mid,))
