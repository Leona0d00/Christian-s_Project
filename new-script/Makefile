# Makefile — Chinese Real-Time Speech-to-Text (Sherpa-ONNX / C)
#
# Works on: Windows cmd.exe / Git Bash / Linux
#
# Usage:
#   mingw32-make          Build all
#   mingw32-make test     Run tests
#   mingw32-make clean    Remove build artifacts

# === Compiler ===
CC       = gcc
CXX      = g++

# === Directories ===
SRC_DIR    = src
BIN_DIR    = build
SHERPA_PKG = deps/sherpa-onnx/sherpa-onnx-v1.13.4-win-x64-shared-MD-Release-no-tts
INCLUDE    = -Iinclude -I$(SHERPA_PKG)/include
LIB_DIR    = -L$(SHERPA_PKG)/lib
LDLIBS     = -lsherpa-onnx-c-api

CFLAGS     = -std=c11 -Wall -O2 $(INCLUDE)
LDFLAGS    = $(LIB_DIR) $(LDLIBS)

# === Source files ===
SRCS = $(SRC_DIR)/core.c $(SRC_DIR)/corrector.c $(SRC_DIR)/audio.c \
       $(SRC_DIR)/pipeline.c $(SRC_DIR)/edge.c $(SRC_DIR)/utils.c \
       $(SRC_DIR)/api.c

OBJS     = $(patsubst $(SRC_DIR)/%.c,$(BIN_DIR)/%.o,$(SRCS))
MAIN_OBJ = $(BIN_DIR)/main.o
CLI_EXE  = $(BIN_DIR)/transcribe_cli.exe
DLL_FILE = $(BIN_DIR)/transcribe.dll

# === Phony targets ===
.PHONY: all test clean help

# ============================================================
all: $(CLI_EXE) $(DLL_FILE) copy-dlls
	@echo ========================================
	@echo  Build complete!
	@echo   $(CLI_EXE)
	@echo   $(DLL_FILE)
	@echo ========================================
	@echo
	@echo Run:  $(subst /,\\,$(CLI_EXE)) -v ..\test.wav
	@echo Test: mingw32-make test

copy-dlls:
	@echo Copying runtime DLLs...
	-cp "$(SHERPA_PKG)/lib/"*.dll "$(BIN_DIR)/"
	-copy /y "$(subst /,\,$(SHERPA_PKG))\lib\*.dll" "$(subst /,\,$(BIN_DIR))\\"
	@echo Done.

# ============================================================
$(CLI_EXE): $(OBJS) $(MAIN_OBJ)
	-"$(CXX)" -o "$@" $^ $(LDFLAGS)

$(DLL_FILE): $(OBJS)
	-"$(CXX)" -shared -o "$@" $^ $(LDFLAGS) \
		-Wl,--out-implib,$(BIN_DIR)/transcribe.lib

$(BIN_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/%.h
	-mkdir "$(subst /,\,$(BIN_DIR))" 2>/dev/null
	-mkdir -p $(BIN_DIR) 2>/dev/null
	"$(CC)" $(CFLAGS) -DTRANSCRIBE_EXPORTS -c -o "$@" "$<"

$(BIN_DIR)/main.o: main.c
	-mkdir "$(subst /,\,$(BIN_DIR))" 2>/dev/null
	-mkdir -p $(BIN_DIR) 2>/dev/null
	"$(CC)" $(CFLAGS) -c -o "$@" "$<"

$(BIN_DIR)/api.o: $(SRC_DIR)/api.c
	-mkdir "$(subst /,\,$(BIN_DIR))" 2>/dev/null
	-mkdir -p $(BIN_DIR) 2>/dev/null
	"$(CC)" $(CFLAGS) -DTRANSCRIBE_EXPORTS -c -o "$@" "$<"

# ============================================================
test: $(DLL_FILE)
	@echo Running tests...
	@"D:/Miniconda/envs/christian/python.exe" -u test/test.py

# ============================================================
clean:
	-rm -f $(BIN_DIR)/*.o $(BIN_DIR)/*.exe $(BIN_DIR)/*.dll $(BIN_DIR)/*.lib 2>/dev/null
	-del /q "$(subst /,\,$(BIN_DIR))\*.o" "$(subst /,\,$(BIN_DIR))\*.exe" "$(subst /,\,$(BIN_DIR))\*.dll" "$(subst /,\,$(BIN_DIR))\*.lib"
	@echo Cleaned.

help:
	@echo Targets:
	@echo "  mingw32-make         Build transcribe_cli.exe + transcribe.dll"
	@echo "  mingw32-make test    Run Python test suite"
	@echo "  mingw32-make clean   Remove build artifacts"
	@echo ""
	@echo "Compiler: $(CC)"
	@echo "Sherpa:   $(SHERPA_PKG)"
