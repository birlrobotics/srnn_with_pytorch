

Features in Binary SVM Format 
(used as input to the learning and inference code provided at https://github.com/hemakoppula/human_activity_labeling/tree/master/svm_struct_learning)

1) features_binary_svm_format.tar.gz  

- Each activity # has one file for node and edge features within a segment and one file for the termporal edge features between two given features.
Example: Lets say activity id a1 has 3 segments in total, then there are a total of 5 feature files: 
a1_1.txt  - has object node features, skeleton node features, object-object edge features and skeleton-object edge features of segment 1
a1_2.txt  - same features as above for segment 2
a1_3.txt  - same features as above for segment 3
a1_1_2.txt - temporal object edge features and temporal skeleton edge features for segment 1 and 2
a1_2_3.txt - same features as above for segment 2 and 3

Format of a1_1.txt : 
- first line: N1 E1 E2 K1 K2 SN 
  N1 - number of object nodes
  E1 - number of object object edges
  E2 - number of skeleton object edges
  K1 - total number of affordance classes
  K2 - total number of sub-activity classes
  FN - segment number
- folowing N1 lines: object node features 
  <affordance_class> <object_id> <feature_num>:<feature_value> <feature_num>:<feature_value> <feature_num>:<feature_value> ...   
- following 1 line : skeleton node features 
  <sub-activity_class> <skel_id> <feature_num>:<feature_value> <feature_num>:<feature_value> <feature_num>:<feature_value> ...
- following E1 lines: object-object edge features  
  <affordance_class_1> <affordance_class_2> <object_id_1> <object_id_2> <feature_num>:<feature_value> <feature_num>:<feature_value> <feature_num>:<feature_value> ...
- following E2 lines: skeleton-object edge features
  <affordance_class> <sub-activity_class> <object_id> <feature_num>:<feature_value> <feature_num>:<feature_value> <feature_num>:<feature_value> ... 


Format of a1_1_2.txt :
- first line: E3 E4 SN1 SN2
  E3 - number of temporal object-object edges
  E4 - number of temporal skeleton-skeleton edges
  SN1 - segment number 1
  SN2 - segment number 2
- following E3 lines: temporal object edge features 
  <affordance_class_sn1> <affordance_class_sn2> <object_id> <feature_num>:<feature_value> <feature_num>:<feature_value> <feature_num>:<feature_value> ...
- following E4 lines: temporal skeleton edge features
  <sub-activity_class_sn1> <sub-activity_class_sn2> <skel_id> <feature_num>:<feature_value> <feature_num>:<feature_value> <feature_num>:<feature_value> ...

2) segments_svm_format.tar.gz

- Each activty # has one file specifying paths to the segment and temporal feature files described above. Path to these files are given as input to the svm learning and inference code. 

Format: 
- first line : N1 N2
  N1 - number of segment feature files
  N2 - number of temporal feature files
- following N1 lines - paths to the segment feature files
- following N2 line - paths to the segment feature files

Replace "path_to_file" in the files to the correct path after download. 
