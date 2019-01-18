all:
	gcc neural.c -std=c99 -lm -lpthread -o neural.o

debug:
	gcc neural.c -g -Wall -Wextra -std=c99 -lm -lpthread -o neural.o