from __future__ import annotations

import os

from pymol.Qt import QtCore, QtWidgets

from pymol.ai.provider_key_store import (
    ProviderKeyStoreError,
    clear_saved_key_and_loaded_env_if_needed,
    get_status,
    resolve_validation_model,
    save_key,
    validate_key_live,
)
from pymol.ai.providers import (
    API_STYLES,
    ENV_CUSTOM_ADDON,
    ENV_CUSTOM_API_STYLE,
    ENV_CUSTOM_NAME,
    ENV_CUSTOM_URL,
    get_provider_spec,
    load_preferred_model,
    provider_menu_entries,
    save_custom_overrides,
    set_active_provider,
)


class _KeyValidationWorker(QtCore.QObject):
    finished = QtCore.Signal(bool, str)

    def __init__(self, provider_id: str, key: str, model: str, timeout_sec: float):
        super().__init__()
        self._provider_id = provider_id
        self._key = str(key or "").strip()
        self._model = str(model or "").strip()
        self._timeout_sec = float(timeout_sec)

    @QtCore.Slot()
    def run(self):
        try:
            validate_key_live(
                self._provider_id,
                self._key,
                model=self._model,
                timeout_sec=self._timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))
            return
        self.finished.emit(True, "API key is valid.")


class AiProviderDialog(QtWidgets.QDialog):
    """ONLYOFFICE-inspired LLM provider settings: name/url/key/addon/api_style."""

    def __init__(self, parent=None, *, runtime=None, on_changed=None):
        super().__init__(parent)
        self._runtime = runtime
        self._on_changed = on_changed
        self._worker = None
        self._worker_thread = None

        self.setWindowTitle("LLM Providers")
        self.setModal(True)
        self.resize(560, 320)

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form = QtWidgets.QFormLayout()
        self.provider_combo = QtWidgets.QComboBox(self)
        for pid, name in provider_menu_entries():
            self.provider_combo.addItem("%s (%s)" % (name, pid), pid)
        form.addRow("Provider", self.provider_combo)

        self.custom_name = QtWidgets.QLineEdit(self)
        self.custom_url = QtWidgets.QLineEdit(self)
        self.custom_addon = QtWidgets.QLineEdit(self)
        self.custom_addon.setPlaceholderText("e.g. v1")
        self.api_style_combo = QtWidgets.QComboBox(self)
        for style in API_STYLES:
            self.api_style_combo.addItem(style, style)
        form.addRow("Custom name", self.custom_name)
        form.addRow("Base URL", self.custom_url)
        form.addRow("URL addon", self.custom_addon)
        form.addRow("API style", self.api_style_combo)

        self.key_input = QtWidgets.QLineEdit(self)
        self.key_input.setEchoMode(QtWidgets.QLineEdit.PasswordEchoOnEdit)
        self.key_input.setPlaceholderText("Enter API key for selected provider")
        form.addRow("API Key", self.key_input)
        layout.addLayout(form)

        buttons_row = QtWidgets.QHBoxLayout()
        self.apply_provider_button = QtWidgets.QPushButton("Use Provider", self)
        self.save_button = QtWidgets.QPushButton("Save Key", self)
        self.clear_button = QtWidgets.QPushButton("Clear Key", self)
        self.test_button = QtWidgets.QPushButton("Test", self)
        self.close_button = QtWidgets.QPushButton("Close", self)
        buttons_row.addWidget(self.apply_provider_button)
        buttons_row.addWidget(self.save_button)
        buttons_row.addWidget(self.clear_button)
        buttons_row.addWidget(self.test_button)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self.close_button)
        layout.addLayout(buttons_row)

        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.apply_provider_button.clicked.connect(self._on_apply_provider)
        self.save_button.clicked.connect(self._on_save)
        self.clear_button.clicked.connect(self._on_clear)
        self.test_button.clicked.connect(self._on_test)
        self.close_button.clicked.connect(self.accept)

        current = getattr(runtime, "provider", None) if runtime is not None else None
        if current:
            idx = self.provider_combo.findData(current)
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
        self._on_provider_changed()
        self._refresh_status()

    def _selected_provider_id(self) -> str:
        data = self.provider_combo.currentData()
        return str(data or "openrouter")

    def _on_provider_changed(self):
        pid = self._selected_provider_id()
        is_custom = pid == "custom"
        for widget in (self.custom_name, self.custom_url, self.custom_addon, self.api_style_combo):
            widget.setEnabled(is_custom)
        spec = get_provider_spec(pid)
        if is_custom:
            self.custom_name.setText(spec.name)
            self.custom_url.setText(spec.url)
            self.custom_addon.setText(spec.addon)
            style_idx = self.api_style_combo.findData(spec.api_style)
            if style_idx >= 0:
                self.api_style_combo.setCurrentIndex(style_idx)
        self._refresh_status()

    def _persist_custom_fields(self):
        if self._selected_provider_id() != "custom":
            return
        name = str(self.custom_name.text() or "").strip() or "Custom URL"
        url = str(self.custom_url.text() or "").strip()
        addon = str(self.custom_addon.text() or "").strip()
        style = str(self.api_style_combo.currentData() or "openai_compat")
        os.environ[ENV_CUSTOM_NAME] = name
        os.environ[ENV_CUSTOM_URL] = url
        os.environ[ENV_CUSTOM_ADDON] = addon
        os.environ[ENV_CUSTOM_API_STYLE] = style
        save_custom_overrides(name=name, url=url, addon=addon, api_style=style)

    def _status_text(self) -> str:
        pid = self._selected_provider_id()
        status = get_status(pid)
        spec = get_provider_spec(pid)
        if status.source == "env" and status.has_key:
            source = "environment"
        elif status.source == "saved" and status.has_key:
            source = "saved keychain"
        else:
            source = "not set"
        return (
            "Provider: %s (%s)\n"
            "API style: %s\n"
            "Base URL: %s\n"
            "Current key source: %s\n"
            "Current key: %s\n"
            "System keychain: %s"
        ) % (
            spec.name,
            spec.id,
            spec.api_style,
            spec.effective_base_url() or "(empty)",
            source,
            status.masked_key or "(none)",
            "available" if status.keyring_available else "unavailable",
        )

    def _refresh_status(self):
        self.status_label.setText(self._status_text())

    def _notify_changed(self):
        callback = self._on_changed
        if callable(callback):
            callback()

    def _on_apply_provider(self):
        self._persist_custom_fields()
        pid = set_active_provider(self._selected_provider_id())
        runtime = self._runtime
        if runtime is not None and hasattr(runtime, "set_provider"):
            runtime.set_provider(pid, emit_notice=True, reset_model=True)
        self._notify_changed()
        self._refresh_status()
        QtWidgets.QMessageBox.information(
            self,
            "LLM Provider",
            "Active provider set to %s. OpenRouter remains available as a rollback option." % (pid,),
        )

    def _on_save(self):
        self._persist_custom_fields()
        pid = self._selected_provider_id()
        key = str(self.key_input.text() or "").strip()
        if not key:
            QtWidgets.QMessageBox.warning(self, "Save API Key", "Please enter a non-empty API key.")
            return
        try:
            save_key(pid, key)
        except ProviderKeyStoreError as exc:
            QtWidgets.QMessageBox.critical(self, "Save API Key", str(exc))
            return
        spec = get_provider_spec(pid)
        os.environ[spec.key_env] = key
        os.environ[spec.key_source_env] = "saved_keyring"
        # Saving a provider key also activates that provider and persists the choice.
        set_active_provider(pid)
        runtime = self._runtime
        if runtime is not None and hasattr(runtime, "set_provider"):
            runtime.set_provider(pid, emit_notice=True, reset_model=True)
        self.key_input.clear()
        self._notify_changed()
        self._refresh_status()
        QtWidgets.QMessageBox.information(
            self,
            "Save API Key",
            "API key saved and active provider set to %s." % (pid,),
        )

    def _on_clear(self):
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Clear API Key",
            "Delete the saved API key for this provider from system keychain?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        pid = self._selected_provider_id()
        try:
            env_cleared = clear_saved_key_and_loaded_env_if_needed(pid)
        except ProviderKeyStoreError as exc:
            QtWidgets.QMessageBox.critical(self, "Clear API Key", str(exc))
            return
        self.key_input.clear()
        self._notify_changed()
        self._refresh_status()
        msg = "Saved API key cleared."
        if env_cleared:
            msg += " Active in-process key was also cleared."
        QtWidgets.QMessageBox.information(self, "Clear API Key", msg)

    def _set_testing_state(self, active: bool):
        for button in (
            self.apply_provider_button,
            self.save_button,
            self.clear_button,
            self.test_button,
            self.close_button,
        ):
            button.setEnabled(not active)

    def _on_test(self):
        if self._worker_thread is not None:
            return
        self._persist_custom_fields()
        pid = self._selected_provider_id()
        spec = get_provider_spec(pid)
        key = str(self.key_input.text() or "").strip() or str(os.getenv(spec.key_env) or "").strip()
        if not key:
            QtWidgets.QMessageBox.warning(self, "Test API Key", "No API key is available to test.")
            return
        model = ""
        runtime = self._runtime
        if runtime is not None and str(getattr(runtime, "provider", "") or "") == pid:
            model = str(getattr(runtime, "model", "") or "")
        model = resolve_validation_model(
            pid,
            model or load_preferred_model(pid) or spec.default_model,
        )
        self._set_testing_state(True)
        worker = _KeyValidationWorker(pid, key, model, timeout_sec=10.0)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_test_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._worker = worker
        self._worker_thread = thread
        thread.start()

    @QtCore.Slot(bool, str)
    def _on_test_finished(self, ok: bool, message: str):
        self._worker = None
        self._worker_thread = None
        self._set_testing_state(False)
        if ok:
            QtWidgets.QMessageBox.information(self, "Test API Key", "API key validation succeeded.")
            return
        QtWidgets.QMessageBox.warning(
            self,
            "Test API Key",
            str(message or "API key validation failed."),
        )
