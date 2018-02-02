# Point to includes of local directories
CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=main.o add.o 
SEQ_FILES=sequential.o

all: $(O_FILES)
	$(CC) -o PA1 $(O_FILES)

main.o: ../src/main.cu
	$(CC) -c ../src/main.cu -o main.o $(INCLUDES)

add.o: ../src/add.cu
	$(CC) -c ../src/add.cu -o add.o $(INCLUDES)

sequential.o: ../src/sequential.cu
	$(CC) -c ../src/sequential.cu -o sequential.o $(INCLUDES)
	$(CC) -o Seq $(SEQ_FILES)

clean:
	rm -f *.o
	rm -f *~
