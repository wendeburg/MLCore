all:
	g++-13 -g -std=c++2b -o programa main.cpp -I./include/

release:
	g++-13 -o3 -std=c++2b -o programa main.cpp -I./include/

clean:
	rm programa