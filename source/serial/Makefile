# Makefile for the serial Smith-Waterman baseline
CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11
TARGET  = serial_sw

all: $(TARGET)

$(TARGET): serial_sw.c sw_common.h
	$(CC) $(CFLAGS) -o $(TARGET) serial_sw.c

run: $(TARGET)
	./$(TARGET) 8000 8000 12345

clean:
	rm -f $(TARGET)

.PHONY: all run clean
