#! /usr/bin/env bash

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

flag=0
reqd_xtns=(avx512cd avx512bw avx512dq avx512f avx512vl amx_tile amx_bf16 amx_int8)
for i in "${reqd_xtns[@]}"
  do 
    if [[ ! $cpuxtns =~ $i ]]
      then 
        run "[Error] : Intel® AVX-512 extensions(fp32), AMX are not available :: ($i)"
      else
        flag=1
    fi
done

if [ "$flag" == "1" ]; then
   run "[Done]: Success, the platform supports AVX-512(fp32), AMX instructions"
fi

flag=0
reqd_xtns=(avx512_vnni)
for i in "${reqd_xtns[@]}"
  do 
    if [[ ! $cpuxtns =~ $i ]]
      then 
        run "[Error] : Intel® AVX-512 VNNI (int8) extensions are not available :: ($i)"
      else
        flag=1
      fi
done

if [ "$flag" == "1" ]; then
  flag=0
  run "[Done]: Success, the platform supports AVX-512 VNNI (int8) instructions"
fi 

flag=0
reqd_xtns=(amx_tile)
for i in "${reqd_xtns[@]}"
  do 
    if [[ ! $cpuxtns =~ $i ]]
      then 
        run "[Error] : Intel® AMX(bf16, int8) extensions are not available :: ($i)"
      else
        flag=1
    fi
done

if [ "$flag" == "1" ]; then
   run "[Done]: Success, the platform supports AMX(bf16, int8) instructions"
fi
