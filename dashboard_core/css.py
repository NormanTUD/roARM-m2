CSS = """
Screen {
    background: black;
    color: #00ff00;
}

#main-container {
    height: 1fr;
}

TabbedContent {
    height: 1fr;
}

TabPane {
    height: 1fr;
}

.tab-content {
    height: 1fr;
    padding: 1;
}

.tab-content > Horizontal {
    height: 1fr;
}

.tab-content > Horizontal > Vertical {
    width: 1fr;
    height: 1fr;
}

.arm-view {
    border: solid $primary;
    height: 32;
    min-height: 20;
    padding: 0 1;
    overflow: hidden;
    border: solid #00ff00;
    background: black;
}

#teach-log, #play-log, #calibrate-log, #servo-log {
    border: solid #005500;
    background: #0a0a0a;
}

.joint-display {
    border: solid #00aa00;
    background: #050505;
}

Header {
    background: #003300;
    color: #00ff00;
}

Footer {
    background: #001a00;
    color: #00aa00;
}

.status-bar {
    background: #001100;
    dock: bottom;
    height: 3;
    background: $panel;
    padding: 0 2;
    layout: horizontal;
}

.status-bar Label {
    width: auto;
    margin: 0 1;
}

#status-activity {
    width: auto;
    min-width: 35;
    margin: 0 1;
    color: $warning;
}

.recording-timer {
    height: 1;
    margin: 0 1;
    color: $error;
}

.joint-display {
    height: 3;
    border: solid $secondary;
    padding: 0 1;
}

.control-buttons {
    height: 3;
    align: center middle;
}

.control-buttons Button {
    margin: 0 1;
}

.info-panel {
    border: solid $success;
    height: auto;
    max-height: 12;
    padding: 1;
}

#teach-log {
    height: 1fr;
    min-height: 5;
    border: solid $primary;
}

#play-log {
    height: 1fr;
    min-height: 5;
    border: solid $primary;
}

#calibrate-log {
    height: 1fr;
    min-height: 5;
    border: solid $primary;
}

#servo-log {
    height: 1fr;
    min-height: 5;
    border: solid $primary;
}

.btn-record {
    background: $error;
    color: white;
}

.btn-play {
    background: $success;
    color: white;
}

.btn-stop {
    background: $warning;
    color: black;
}

DataTable {
    height: 1fr;
    min-height: 5;
}

RichLog {
    scrollbar-gutter: stable;
}

#teach-left {
    width: 2fr;
    height: 1fr;
}

#teach-right {
    width: 1fr;
    height: auto;
    max-height: 100%;
}

.servo-control-panel {
    height: auto;
    border: solid $accent;
    padding: 1;
    margin: 0 0 1 0;
}

.servo-slider-row {
    height: 3;
    layout: horizontal;
}

.servo-slider-row Label {
    width: 12;
}

.servo-slider-row Input {
    width: 12;
}

#log-search-input {
    dock: top;
    height: 3;
    margin: 0 0 1 0;
}

#roarm-file-viewer {
    width: 1fr;
    height: 1fr;
    border: solid $accent;
    padding: 0 1;
    overflow-y: auto;
}

#log-viewer {
    height: 1fr;
    border: solid $primary;
}

.log-filter-bar {
    height: 3;
    layout: horizontal;
    dock: top;
}

.log-filter-bar Input {
    width: 1fr;
}

.log-filter-bar Button {
    width: auto;
    margin: 0 1;
}

.recording-active { border: heavy red; }
"""
