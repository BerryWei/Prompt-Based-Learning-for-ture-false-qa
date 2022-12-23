#!/bin/bash
#SBATCH --job-name="HW3"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:10
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt
#SBATCH --chdir=.
###SBATCH --test-only

timer_start=`date "+%Y-%m-%d %H:%M:%S"`

bash train.sh

timer_end=`date "+%Y-%m-%d %H:%M:%S"`

duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "Start Time： $timer_start"
echo "End Time： $timer_end"
echo "Duration： $duration"
