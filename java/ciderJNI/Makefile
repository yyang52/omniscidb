CC=gcc
CFLAGS=-c -Wall -fPIC -I/home/chengxu1/miniconda3/envs/omnisci-dev/include/ -I/home/chengxu1/miniconda3/envs/omnisci-dev/include/linux
LDFLAGS=-fPIC -shared

SOURCES_DIR=src/main/native
OBJECTS_DIR=target/native
EXECUTABLE=target/classes/libciderjni.so

SOURCES=$(shell find '$(SOURCES_DIR)' -type f -name '*.cpp')
OBJECTS=$(SOURCES:$(SOURCES_DIR)/%.cpp=$(OBJECTS_DIR)/%.o)

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

$(OBJECTS): $(SOURCES)
	mkdir -p $(OBJECTS_DIR)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS_DIR) $(EXECUTABLE)
