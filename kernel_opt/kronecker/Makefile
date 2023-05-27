include Makefile.inc

LIBS+=-L$(GRAPH500) -lgraph500
INC+=-I$(GRAPH500)

OBJS = \
	graph.o

EXE = \
  main

# clear out all suffixes
.SUFFIXES:
# list only those we use
.SUFFIXES: .o .c .cpp

# some suffix rules
.c.o:
	$(CC) $(CCFLAGS) $(INC) -c $<

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INC) -c $<

.PHONY: all
all: graph.so

.PHONY: clean
clean:
	rm -rf main.o $(OBJS) $(EXE) graph.so
	git status

.PHONY: distclean
distclean: clean
	make -C $(GRAPH500) clean
	git status

$(GRAPH500)/libgraph500.a:
	make -C $(GRAPH500) libgraph500.a

main: graph.o main.o $(GRAPH500)/libgraph500.a
	$(CC) $(CCFLAGS) $(LDFLAGS) -o $@ main.o $(OBJS) $(LIBS)

main.o: main.c graph.h

graph.o: graph.c $(GRAPH500)/make_graph.h $(GRAPH500)/utils.h

graph.so: graph.o $(GRAPH500)/libgraph500.a
	$(CC) -shared $(CCFLAGS) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)
