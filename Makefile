CC = gcc
CFLAGS = -O2 -pthread -lm
SRC = llama2.c utilities.c
OBJ = $(SRC:.c=.o)
TARGET = llama2
MODEL_URL = https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
TOKENIZER_URL = https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin
MODEL_FILE = model.bin
TOKENIZER_FILE = tokenizer.bin

$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.c
	$(CC) -c $< $(CFLAGS)

download:
	wget -O $(MODEL_FILE) $(MODEL_URL)
	wget -O $(TOKENIZER_FILE) $(TOKENIZER_URL)

clean:
	rm -f $(OBJ) $(TARGET)

cleanbin:
	rm -f $(MODEL_FILE) $(TOKENIZER_FILE)

