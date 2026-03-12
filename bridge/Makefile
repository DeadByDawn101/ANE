# ANE Bridge — builds libane_bridge.dylib
# Requires: Apple Silicon Mac, Xcode Command Line Tools

CC      = clang
CFLAGS  = -O2 -arch arm64
OBJCFLAGS = $(CFLAGS) -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML

SRCS    = ane_bridge.m ane_memory.m ane_dmabuf.m
HEADERS = ane_bridge.h
OUTPUT  = libane_bridge.dylib

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): $(SRCS) $(HEADERS)
	$(CC) $(OBJCFLAGS) $(FRAMEWORKS) \
		-dynamiclib \
		-install_name @rpath/$(OUTPUT) \
		-o $(OUTPUT) $(SRCS)
	@echo "Built: $(OUTPUT)"

clean:
	rm -f $(OUTPUT) *.o
