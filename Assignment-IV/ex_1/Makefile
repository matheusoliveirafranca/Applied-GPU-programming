CXX = pgc++
SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:.cpp=.o)
BIN = ./bin
TARGET = miniWeather

LDFLAGS = -Wl,-rpath -Wl,/pdc/vol/openmpi/3.0/gcc/6.2.0/lib -Wl,--enable-new-dtags -L/pdc/vol/openmpi/3.0/gcc/6.2.0/lib -lmpi -lpthread -lm
CFLAGS = -I./include -I/pdc/vol/openmpi/3.0/gcc/6.2.0/include -pg -O3
OPENACC_FLAGS = -acc -ta=tesla:cc30,nvidia,lineinfo -Minfo=accel

all: dir $(BIN)/$(TARGET)

dir: ${BIN}

${BIN}:
	mkdir -p $(BIN)

%.o: %.cpp
	$(CXX) $(OPENACC_FLAGS) $(CFLAGS) -c -o $@ $<

$(BIN)/$(TARGET): $(OBJ) lib/libpnetcdf.a
	$(CXX) $(OPENACC_FLAGS) $(LDFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f $(OBJ) $(BIN)/$(TARGET)

#pgc++ -I/include -O3 -acc -ta=tesla:cc3x,nvidia,lineinfo,debug -Minfo=accel -o miniWeather_mpi_openacc.out miniWeather_mpi_openacc.cpp /cfs/klemming/nobackup/w/wdchien/tegner_local/lib/libpnetcdf.a -I/cfs/klemming/nobackup/w/wdchien/tegner_local/include -I/pdc/vol/openmpi/3.0/gcc/6.2.0/include -lpthread -Wl,-rpath -Wl,/pdc/vol/openmpi/3.0/gcc/6.2.0/lib -Wl,--enable-new-dtags -L/pdc/vol/openmpi/3.0/gcc/6.2.0/lib -lmpi -g
