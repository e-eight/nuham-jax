// Decode a .gz file to stdout exactly the way imsrg++'s ReadWrite.cc builds its reader:
// std::ifstream -> boost gzip_decompressor -> filtering_istream. Reports bytes and stream state.
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char** argv)
{
  std::ifstream infilegz(argv[1], std::ios_base::in | std::ios_base::binary);
  boost::iostreams::filtering_istream infile;
  infile.push(boost::iostreams::gzip_decompressor());
  infile.push(infilegz);
  std::vector<char> buf(1 << 20);
  unsigned long long n = 0;
  try {
    while (infile.read(buf.data(), buf.size()) || infile.gcount() > 0) {
      std::cout.write(buf.data(), infile.gcount());
      n += infile.gcount();
    }
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << "\n";
    return 2;
  }
  std::cerr << "bytes " << n << " eof=" << infile.eof() << " bad=" << infile.bad() << " fail=" << infile.fail() << "\n";
  return infile.bad() ? 1 : 0;
}
