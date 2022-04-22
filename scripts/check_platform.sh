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

cpuxtns=$(lscpu | grep -i "avx512")

run "[init]: checking if AVX512(FP32) instructions are available"
reqd_xtns=(avx512cd avx512bw avx512dq avx512f avx512vl)

for i in "${reqd_xtns[@]}"
  do 
    if [[ ! $cpuxtns =~ $i ]]
      then 
        run "[Error] : Intel® AVX-512 extensions are not available :: ($i)"
        exit
      else
        run "[Done]: Success, the platform supports AVX-512 instructions"
    fi
done

run "[init]: checking if AVX-512 VNNI( int8) instructions are available"
reqd_xtns=(avx512cd avx512bw avx512dq avx512f avx512vl)

for i in "${reqd_xtns[@]}"
  do 
    if [[ ! $cpuxtns =~ $i ]]
      then 
        run "[Error] : Intel® AVX-512 VNNI( int8) extensions are not available :: ($i)"
        exit
      else
        run "[Done]: Success, the platform supports AVX-512 VNNI( int8) instructions"
    fi
  done

run "[init]: checking if AMX(bf16, int8) instructions are available"
reqd_xtns=(amx_tile)

for i in "${reqd_xtns[@]}"
  do 
    if [[ ! $cpuxtns =~ $i ]]
      then 
        run "[Error] : Intel® AMX(bf16, int8) extensions are not available :: ($i)"
        exit
      else
        run "[Done]: Success, the platform supports AMX(bf16, int8) instructions"
    fi
  done
