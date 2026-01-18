CXX = g++
CXXFLAGS = -O3 -shared -fPIC -Wall -Wextra -std=c++11


TARGET = market_engine.so
SRC = strategy_engine.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
