# my stc ntcir12 task code

s1 = similarity (test_post, database_post)
s2 = similarity (database_post, database_response)
s3 = similarity (test_post, database_response)

final_score = a * s1 + b * s2 + c * s3

use train set and svm_rank to train parameters a, b and c

and finally sorting final score and choose top 10 as my response.

http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings12/NTCIR/toc_ntcir.html

http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings12/pdf/ntcir/STC/16-NTCIR12-STC-LuoC.pdf


![image](https://github.com/luochuwei/stc_ntcir12_code/raw/master/data/POSTER-2.jpg)
