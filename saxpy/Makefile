ROCM_PATH ?= /opt/rocm
SRC_PATH	     = ./src
DRIVER_FILE = $(SRC_PATH)/driver.cpp
KERNEL_FILE = $(SRC_PATH)/kernel.cpp
COMMON_FILE  = ../common/common.cpp
OBJS_MEASURE = $(BUILD_PATH)/driver.o $(BUILD_PATH)/kernel.o $(BUILD_PATH)/common.o
BUILD_PATH   = ./build

KERNEL = BASIC
CC = hipcc
CFLAGS = -O3 -D $(KERNEL) -fopenmp

INC_DIRS   := $(shell find ./include ../common -type d)
INC_FLAGS  := $(addprefix -I,$(INC_DIRS))

CFLAGS += $(INC_FLAGS) 

.PHONY: $(BUILD_PATH)/kernel.o $(BUILD_PATH)/driver.o

all: saxpy 

$(BUILD_PATH)/driver.o: $(DRIVER_FILE)
	@echo Building $@ . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_PATH)/kernel.o: $(KERNEL_FILE)
	@echo Building $@ . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_PATH)/common.o: $(COMMON_FILE)
	@echo Building $@ . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

saxpy: $(OBJS_MEASURE)
	@echo Building $@ . . .
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS) 
	@roc-obj -d $@

clean :
	rm -rf ./build saxpy