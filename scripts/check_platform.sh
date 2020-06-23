set -e
set -u
set -o pipefail

run()
{
  line_length=$(echo "$@" | awk '{print length}')
  printf "%$((line_length+35))s\n" |tr " " "="
  printf "$(date) -- %s"
  printf "%s\n" "$@"
  printf "%$((line_length+35))s\n" |tr " " "="
}
reqd_xtns=(avx512cd avx512bw avx512dq avx512f avx512vl)
cpuxtns=$(lscpu | grep -i "avx512")
for i in "${reqd_xtns[@]}"
  do 
    if [[ ! $cpuxtns =~ $i ]]
      then 
        run "[Error] : Intel® AVX-512 extensions required by DLRS not available :: ($i)"
        exit
    fi
  done
run "[Done]: Success, the platform supports AVX-512 instructions"
