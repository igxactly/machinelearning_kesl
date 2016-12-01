CC=icc
ELF=mnist

ROOT_PATH= .

LIB_PATH = $(ROOT_PATH)/libs/

EVALUATOR_PATH= ./evaluator
FORWARDER_PATH= ./forwarder
LEARNER_PATH= ./learner
READER_PATH= ./reader

# OBJS += $(EVALUATOR_PATH)/
# OBJS += $(FORWARDER_PATH)/
# OBJS += $(LEARNER_PATH)/
# OBJS += $(READER_PATH)/

OBJS += main.o network.o jsmn.o
INCLUDE += $(LIB_PATH)

CFLAGS += -I$(INCLUDE)
CFLAGS += -xMIC-AVX512
#CFLAGS += -g
#CFLAGS += -fopenmp

all: ${ELF}

${ELF}: ${OBJS}
	$(CC) $(CFLAGS) $(OBJS) -lm -o ${ELF}

network.o : ${ROOT_PATH}/network.c
	$(CC) -c ${ROOT_PATH}/network.c $(CFLAGS)

main.o : ${ROOT_PATH}/main.c
	$(CC) -c ${ROOT_PATH}/main.c $(CFLAGS) 

jsmn.o : ${LIB_PATH}/jsmn/jsmn.c
	$(CC) -c ${LIB_PATH}/jsmn/jsmn.c $(CFLAGS) 

clean:
	rm -rf ./*.o  rm ./mnist
