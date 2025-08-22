# downloaded file list from aws bucket on cluster and moved it here
aws s3 ls s3://cb-oncology/p570/s6015/rnaseq/ --recursive > rnaseq_file_list.txt
aws s3 ls s3://cb-oncology/p570/s6015/wes/ --recursive > wes_file_list.txt

cat wes_file_list.txt | egrep 'germline' | tr -s " " | cut -d " " -f 4 | egrep 'fastq.gz$'  > wes_fastq_list_germline.txt

cat wes_file_list.txt | egrep 'tumor-wes' | tr -s " " | cut -d " " -f 4 | egrep 'fastq.gz$'  > wes_fastq_list_tumor.txt



cat wes_fastq_list_tumor.txt | \
    tblkit --noheader tbl_add_header | \
    tblkit col_extract -c c1 -p '_(S[0-9]{2,})_' | \
    tblkit col_extract -c c1 -p '[_/](SA[0-9]+D)_'  | \
    tblkit col_extract -c c1 -p '\b(SH[0-9]+)_'  | \
    tblkit col_rename --map "c1:filepath,c1_captured:sample_number,c1_captured_1:vendor_id,c1_captured_2:lane_id" | head


