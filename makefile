all:
	g++ -g -std=c++2b -o programa main.cpp -I./include/

release:
	g++ -o3 -std=c++2b -o programa main.cpp -I./include/

clean:
	rm programa