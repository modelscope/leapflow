# ── Variables ─────────────────────────────────────────────────────────────────
LEAPFLOW_DATA_DIR ?= $(HOME)/.leapflow
HOST_ROOT     := $(LEAPFLOW_DATA_DIR)/host
HOST_SOCKET   := $(LEAPFLOW_DATA_DIR)/var/host.sock
HOST_PID      := $(LEAPFLOW_DATA_DIR)/var/host.pid
HOST_LOG      := $(LEAPFLOW_DATA_DIR)/var/host.log

# ── OS Host platform detection ────────────────────────────────────────────────
HOST_OS       := $(shell uname -s)

ifeq ($(HOST_OS),Darwin)
  # macOS: direct swiftc (workaround for CLT 26.x SPM manifest bug)
  HOST_SRC      := os_host/darwin/Sources/OSHost
  HOST_SOURCES  := $(shell find $(HOST_SRC) -name '*.swift')
  SWIFTC_FLAGS  := -module-name OSHost -swift-version 5 \
                  -target arm64-apple-macosx14.0 \
                  -sdk $(shell xcrun --show-sdk-path) \
                  -vfsoverlay os_host/darwin/vfs_overlay.yaml
else ifeq ($(HOST_OS),Linux)
  $(info OS Host for Linux: not yet implemented)
else
  $(info OS Host for $(HOST_OS): not yet implemented)
endif

.PHONY: setup sync test brain host swift-build lint \
        host-build host-install host-start host-stop host-restart \
        host-status host-setup host-clean host-dev

setup:  ## Setup scripts permissions and environment
	chmod +x scripts/setup.sh scripts/run.sh
	./scripts/setup.sh

sync:  ## Sync all dependencies
	uv sync --all-extras

test:  ## Run tests
	uv run pytest tests/ -q

lint:  ## Lint source code
	uv run ruff check src/leapflow/ tests/

# LeapFlow CLI (pass PROMPT via ARGS, e.g. make brain ARGS='--mock-host --prompt "hello"')
brain:  ## Start Brain process
	uv run leap $(ARGS)

swift-build:  ## Build OS Host (debug)
ifeq ($(HOST_OS),Darwin)
	@mkdir -p os_host/darwin/.build/debug
	swiftc $(SWIFTC_FLAGS) -g -Onone -o os_host/darwin/.build/debug/OSHost $(HOST_SOURCES)
	@echo "Built: os_host/darwin/.build/debug/OSHost"
else
	@echo "Error: OS Host build not yet supported on $(HOST_OS)" && exit 1
endif

host:  ## Run OS Host in foreground (debug)
	@$(MAKE) swift-build
	LEAPFLOW_BRIDGE_SOCKET=$${LEAPFLOW_BRIDGE_SOCKET:-$(HOST_SOCKET)} os_host/darwin/.build/debug/OSHost

# ── OS Host Service Management ────────────────────────────────────────────────

host-build:  ## Build OS Host (release)
ifeq ($(HOST_OS),Darwin)
	@mkdir -p os_host/darwin/.build/release
	swiftc $(SWIFTC_FLAGS) -O -o os_host/darwin/.build/release/OSHost $(HOST_SOURCES)
	@echo "Built: os_host/darwin/.build/release/OSHost"
else
	@echo "Error: OS Host build not yet supported on $(HOST_OS)" && exit 1
endif

host-install: host-build  ## Build + package as .app bundle + deploy to ~/.leapflow/host/
	@mkdir -p $(HOST_ROOT)/LeapHost.app/Contents/MacOS
	@mkdir -p $(HOST_ROOT)/LeapHost.app/Contents/Resources
	@cp os_host/darwin/.build/release/OSHost $(HOST_ROOT)/LeapHost.app/Contents/MacOS/LeapHost
	@cp os_host/darwin/Resources/Info.plist $(HOST_ROOT)/LeapHost.app/Contents/Info.plist
	@chmod +x $(HOST_ROOT)/LeapHost.app/Contents/MacOS/LeapHost
	@echo "Installed to $(HOST_ROOT)/LeapHost.app"

host-start:  ## Start OS Host service
	@mkdir -p $(LEAPFLOW_DATA_DIR)/var
	$(HOST_ROOT)/LeapHost.app/Contents/MacOS/LeapHost \
		--daemon --socket $(HOST_SOCKET) --pid-file $(HOST_PID) --log-file $(HOST_LOG) &
	@echo "OS Host started"

host-stop:  ## Stop OS Host service
	@if [ -f $(HOST_PID) ]; then \
		kill $$(cat $(HOST_PID)) 2>/dev/null || true; \
		echo "OS Host stopped"; \
	else \
		echo "OS Host not running"; \
	fi

host-restart: host-stop host-start  ## Restart OS Host

host-status:  ## Show OS Host status
	@if [ -f $(HOST_PID) ] && kill -0 $$(cat $(HOST_PID)) 2>/dev/null; then \
		echo "● Running (PID $$(cat $(HOST_PID)))"; \
	else \
		echo "○ Stopped"; \
	fi

host-setup: host-install  ## Full setup: build + install + register launchd
	@python -m leapflow.cli.commands.host _register_launchd 2>/dev/null || \
		echo "Run 'leap host setup' for full LaunchAgent registration"
	@echo "Setup complete. Run 'leap host start' or re-login for auto-start."

host-clean:  ## Remove installed host
	@rm -rf $(HOST_ROOT)/LeapHost.app
	@echo "Host app removed"

host-dev:  ## Run OS Host in dev mode (auto-rebuild on changes)
	uv run leap host dev
