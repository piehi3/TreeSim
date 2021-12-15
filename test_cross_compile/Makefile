# Makefile for PMT test facility
#
# $Id: Makefile 1131 2008-05-22 21:35:42Z olchansk $
# 
# $Log$
#

#VPATH = degauss

CFLAGS   = -g -O2
CXXFLAGS = $(CFLAGS)

#CXX = /home/idwcorni/llvm-mingw-20211002-ucrt-aarch64/bin/aarch64-w64-mingw32-g++.exe#g++
CXX = /home/idwcorni/llvm-mingw-aarch64/bin/aarch64-w64-mingw32-g++
#CXX = g++
#USE_VMICVME=1
USE_GEFVME=1

# support libraries

#LIBS = 

# all: default target for "make"

all:Test

Test: test.o
	$(CXX) -o $@ $(CFLAGS) $^

# DEPRECATED (from test phase)
#feScanNew: $(MIDASLIBS) $(MFE) feScan_new.o 
#	$(CXX) -o $@ $(CFLAGS)  $^ $(MIDASLIBS) $(LIBS) $(VMELIBS)

# DEPRECATED (from test phase)
#feScanNewNew: $(MIDASLIBS) $(MFE) feScan_newnew.o ScanSequence.o
#	$(CXX) -o $@ $(CFLAGS)  $^ $(MIDASLIBS) $(LIBS) $(VMELIBS)

#feDegauss: $(MIDASLIBS) $(MFE) degauss.o feDegauss.o
#	$(CXX) -o $@ $(CFLAGS) $^ $(MIDASLIBS) $(LIBS) $(VMELIBS) -DDEBUG


%.o: %.cxx
	$(CXX) -o $@ -c $< $(CXXFLAGS)

#%.o: %.cpp
#	$(CXX) -o $@ -c $< $(CXXFLAGS)



clean:
	rm -f *.o *.gch *.dSYM Test

# end