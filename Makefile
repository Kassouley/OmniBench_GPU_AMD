ROCM_PATH 	?= /opt/rocm
SRC_PATH	 = ./src
INC_PATH	 = ./include
BUILD_PATH 	 = ./build
BENCH_PATH 	 = ./benchmark
OBJ_PATH 	 = $(BUILD_PATH)/obj
BIN_PATH	 = $(BUILD_PATH)/bin

OPT 	?= NOPT
DIM 	?= ONE_DIM
CC 		 = hipcc
KERNEL 	?= matrixMultiply
CFLAGS	 = -O3 -D KERNEL_NAME=\"$(KERNEL)\" -D OPTIM=\"$(OPT)\" -D $(OPT) -D $(DIM) -fopenmp $(INC_FLAGS)

ifeq ($(OPT), ROCBLAS)
	LFLAGS += -lrocblas
endif
ifeq ($(OPT), TILE)
	CFLAGS += -D TILE_SIZE=$(TILE_SIZE)
endif

ROCPROF_ONLY ?= 0
ifeq ($(ROCPROF_ONLY), 1)
	CFLAGS += -D ROCPROF_ONLY
endif

DIR_COMMON 		   := ./common
SRC_COMMON_MEASURE := $(shell find $(DIR_COMMON)/$(SRC_PATH) -name '*.cpp' -not -name 'main_check.cpp')
OBJ_COMMON_MEASURE := $(addprefix $(DIR_COMMON)/$(OBJ_PATH)/,$(notdir $(SRC_COMMON_MEASURE:.cpp=.o)))

SRC_COMMON_CHECK := $(shell find $(DIR_COMMON)/$(SRC_PATH) -name '*.cpp' -not -name 'main.cpp')
OBJ_COMMON_CHECK := $(addprefix $(DIR_COMMON)/$(OBJ_PATH)/,$(notdir $(SRC_COMMON_CHECK:.cpp=.o)))

DIR_KERNEL 		   := $(BENCH_PATH)/$(KERNEL)
SRC_KERNEL_MEASURE := $(shell find $(DIR_KERNEL) -name '*.cpp' -not -name 'driver_check.cpp')
OBJ_KERNEL_MEASURE := $(addprefix $(DIR_KERNEL)/$(OBJ_PATH)/,$(notdir $(SRC_KERNEL_MEASURE:.cpp=.o)))

SRC_KERNEL_CHECK := $(shell find $(DIR_KERNEL) -name '*.cpp' -not -name 'driver.cpp')
OBJ_KERNEL_CHECK := $(addprefix $(DIR_KERNEL)/$(OBJ_PATH)/,$(notdir $(SRC_KERNEL_CHECK:.cpp=.o)))

INC_DIRS  := $(shell find $(DIR_KERNEL)/$(INC_PATH) $(DIR_COMMON)/$(INC_PATH) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))


all: measure check

$(DIR_COMMON)/$(OBJ_PATH)/%.o: $(DIR_COMMON)/$(SRC_PATH)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Building $@ ..."
	$(CC) -c $< -o $@ $(CFLAGS)

$(DIR_KERNEL)/$(OBJ_PATH)/%.o: $(DIR_KERNEL)/$(SRC_PATH)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Building $@ ..."
	$(CC) -c $< -o $@ $(CFLAGS)

kernel: $(DIR_KERNEL)/$(OBJ_PATH)/kernel.o

measure: $(OBJ_COMMON_MEASURE) $(OBJ_KERNEL_MEASURE)
	@mkdir -p $(DIR_KERNEL)/$(BIN_PATH)
	$(CC) -o $(DIR_KERNEL)/$(BIN_PATH)/$@ $^ $(CFLAGS) $(LFLAGS)

check: $(OBJ_COMMON_CHECK) $(OBJ_KERNEL_CHECK)
	@mkdir -p $(DIR_KERNEL)/$(BIN_PATH)
	$(CC) -o $(DIR_KERNEL)/$(BIN_PATH)/$@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(DIR_COMMON)/build/
	rm -rf $(DIR_KERNEL)/build/
	