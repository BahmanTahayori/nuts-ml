Search.setIndex({docnames:["contributions","faq","index","installation","introduction","nutsml","nutsml.examples","nutsml.examples.autoencoder","nutsml.examples.cifar","nutsml.examples.mnist","overview","tutorial/augment_images","tutorial/batching","tutorial/cifar10_example","tutorial/configuration","tutorial/custom_nuts","tutorial/introduction","tutorial/loading_images","tutorial/logging","tutorial/network","tutorial/plotting","tutorial/reading_samples","tutorial/split_stratify","tutorial/transform_images","tutorial/view_images"],envversion:53,filenames:["contributions.rst","faq.rst","index.rst","installation.rst","introduction.rst","nutsml.rst","nutsml.examples.rst","nutsml.examples.autoencoder.rst","nutsml.examples.cifar.rst","nutsml.examples.mnist.rst","overview.rst","tutorial\\augment_images.rst","tutorial\\batching.rst","tutorial\\cifar10_example.rst","tutorial\\configuration.rst","tutorial\\custom_nuts.rst","tutorial\\introduction.rst","tutorial\\loading_images.rst","tutorial\\logging.rst","tutorial\\network.rst","tutorial\\plotting.rst","tutorial\\reading_samples.rst","tutorial\\split_stratify.rst","tutorial\\transform_images.rst","tutorial\\view_images.rst"],objects:{"":{batcher:[5,0,0,"-"],booster:[5,0,0,"-"],checkpoint:[5,0,0,"-"],cnn_predict:[9,0,0,"-"],cnn_train:[9,0,0,"-"],common:[5,0,0,"-"],config:[5,0,0,"-"],datautil:[5,0,0,"-"],fileutil:[5,0,0,"-"],imageutil:[5,0,0,"-"],logger:[5,0,0,"-"],mlp_precit:[9,0,0,"-"],mlp_predict:[9,0,0,"-"],mlp_train:[9,0,0,"-"],mlp_view_misclassified:[8,0,0,"-"],network:[5,0,0,"-"],nutsml:[5,0,0,"-"],plotter:[5,0,0,"-"],read_images:[9,0,0,"-"],reader:[5,0,0,"-"],stratify:[5,0,0,"-"],transformer:[5,0,0,"-"],view_augmented_images:[8,0,0,"-"],view_data:[8,0,0,"-"],view_train_images:[9,0,0,"-"],viewer:[5,0,0,"-"],write_images:[9,0,0,"-"],writer:[5,0,0,"-"]},"nutsml.batcher":{BuildBatch:[5,1,1,""],Mixup:[5,1,1,""],build_image_batch:[5,3,1,""],build_number_batch:[5,3,1,""],build_one_hot_batch:[5,3,1,""],build_tensor_batch:[5,3,1,""],build_vector_batch:[5,3,1,""]},"nutsml.batcher.BuildBatch":{__init__:[5,2,1,""],__rrshift__:[5,2,1,""],by:[5,2,1,""],input:[5,2,1,""],output:[5,2,1,""]},"nutsml.batcher.Mixup":{__call__:[5,2,1,""]},"nutsml.booster":{Boost:[5,1,1,""],random:[5,3,1,""]},"nutsml.booster.Boost":{__rrshift__:[5,2,1,""]},"nutsml.checkpoint":{Checkpoint:[5,1,1,""]},"nutsml.checkpoint.Checkpoint":{__init__:[5,2,1,""],datapaths:[5,2,1,""],dirs:[5,2,1,""],latest:[5,2,1,""],load:[5,2,1,""],save:[5,2,1,""],save_best:[5,2,1,""]},"nutsml.common":{CheckNaN:[5,1,1,""],ConvertLabel:[5,1,1,""],PartitionByCol:[5,1,1,""],SplitRandom:[5,1,1,""]},"nutsml.common.CheckNaN":{__call__:[5,2,1,""]},"nutsml.common.ConvertLabel":{__call__:[5,2,1,""],__init__:[5,2,1,""]},"nutsml.common.PartitionByCol":{__rrshift__:[5,2,1,""]},"nutsml.common.SplitRandom":{__rrshift__:[5,2,1,""]},"nutsml.config":{Config:[5,1,1,""],load_config:[5,3,1,""]},"nutsml.config.Config":{__init__:[5,2,1,""],isjson:[5,4,1,""],load:[5,2,1,""],save:[5,2,1,""]},"nutsml.datautil":{col_map:[5,3,1,""],group_by:[5,3,1,""],group_samples:[5,3,1,""],isnan:[5,3,1,""],random_downsample:[5,3,1,""],shapestr:[5,3,1,""],shuffle_sublists:[5,3,1,""],upsample:[5,3,1,""]},"nutsml.examples":{autoencoder:[7,0,0,"-"],cifar:[8,0,0,"-"],mnist:[9,0,0,"-"]},"nutsml.examples.autoencoder":{runner:[7,0,0,"-"]},"nutsml.examples.autoencoder.runner":{Diff:[7,1,1,""],create_network:[7,3,1,""],load_samples:[7,3,1,""],predict:[7,3,1,""],train:[7,3,1,""],view:[7,3,1,""]},"nutsml.examples.autoencoder.runner.Diff":{__call__:[7,2,1,""]},"nutsml.examples.cifar":{cnn_classify:[8,0,0,"-"],cnn_train:[8,0,0,"-"],read_images:[8,0,0,"-"],view_augmented_images:[8,0,0,"-"],view_data:[8,0,0,"-"],view_train_images:[8,0,0,"-"],write_images:[8,0,0,"-"]},"nutsml.examples.cifar.cnn_train":{create_network:[8,3,1,""],load_names:[8,3,1,""],load_samples:[8,3,1,""],train:[8,3,1,""]},"nutsml.examples.mnist":{cnn_classify:[9,0,0,"-"],cnn_train:[9,0,0,"-"],mlp_classify:[9,0,0,"-"],mlp_train:[9,0,0,"-"],mlp_view_misclassified:[9,0,0,"-"],read_images:[9,0,0,"-"],view_train_images:[9,0,0,"-"],write_images:[9,0,0,"-"]},"nutsml.examples.mnist.cnn_train":{create_network:[9,3,1,""],load_samples:[9,3,1,""],train:[9,3,1,""]},"nutsml.examples.mnist.mlp_train":{create_network:[9,3,1,""],load_samples:[9,3,1,""],train:[9,3,1,""]},"nutsml.examples.mnist.write_images":{load_samples:[9,3,1,""]},"nutsml.fileutil":{clear_folder:[5,3,1,""],create_filename:[5,3,1,""],create_folders:[5,3,1,""],create_temp_filepath:[5,3,1,""],delete_file:[5,3,1,""],delete_folders:[5,3,1,""],delete_temp_data:[5,3,1,""]},"nutsml.imageutil":{add_channel:[5,3,1,""],annotation2coords:[5,3,1,""],annotation2mask:[5,3,1,""],annotation2pltpatch:[5,3,1,""],arr_to_pil:[5,3,1,""],centers_inside:[5,3,1,""],change_brightness:[5,3,1,""],change_color:[5,3,1,""],change_contrast:[5,3,1,""],change_sharpness:[5,3,1,""],crop:[5,3,1,""],crop_center:[5,3,1,""],crop_square:[5,3,1,""],distort_elastic:[5,3,1,""],enhance:[5,3,1,""],extract_patch:[5,3,1,""],fliplr:[5,3,1,""],flipud:[5,3,1,""],floatimg2uint8:[5,3,1,""],gray2rgb:[5,3,1,""],identical:[5,3,1,""],load_image:[5,3,1,""],mask_choice:[5,3,1,""],mask_where:[5,3,1,""],normalize_histo:[5,3,1,""],occlude:[5,3,1,""],patch_iter:[5,3,1,""],pil_to_arr:[5,3,1,""],polyline2coords:[5,3,1,""],rerange:[5,3,1,""],resize:[5,3,1,""],rgb2gray:[5,3,1,""],rotate:[5,3,1,""],sample_labeled_patch_centers:[5,3,1,""],sample_mask:[5,3,1,""],sample_patch_centers:[5,3,1,""],sample_pn_patches:[5,3,1,""],save_image:[5,3,1,""],set_default_order:[5,3,1,""],shear:[5,3,1,""],translate:[5,3,1,""]},"nutsml.logger":{LogCols:[5,1,1,""],LogToFile:[5,1,1,""]},"nutsml.logger.LogCols":{__init__:[5,2,1,""]},"nutsml.logger.LogToFile":{"delete":[5,2,1,""],__call__:[5,2,1,""],__init__:[5,2,1,""],close:[5,2,1,""]},"nutsml.network":{EvalNut:[5,1,1,""],KerasNetwork:[5,1,1,""],LasagneNetwork:[5,1,1,""],Network:[5,1,1,""],PredictNut:[5,1,1,""],TrainValNut:[5,1,1,""]},"nutsml.network.EvalNut":{__rrshift__:[5,2,1,""]},"nutsml.network.KerasNetwork":{__init__:[5,2,1,""],evaluate:[5,2,1,""],load_weights:[5,2,1,""],predict:[5,2,1,""],print_layers:[5,2,1,""],save_weights:[5,2,1,""],train:[5,2,1,""],validate:[5,2,1,""]},"nutsml.network.LasagneNetwork":{__init__:[5,2,1,""],evaluate:[5,2,1,""],load_weights:[5,2,1,""],predict:[5,2,1,""],print_layers:[5,2,1,""],save_weights:[5,2,1,""],train:[5,2,1,""],validate:[5,2,1,""]},"nutsml.network.Network":{__init__:[5,2,1,""],evaluate:[5,2,1,""],load_weights:[5,2,1,""],predict:[5,2,1,""],print_layers:[5,2,1,""],save_best:[5,2,1,""],save_weights:[5,2,1,""],train:[5,2,1,""],validate:[5,2,1,""]},"nutsml.network.PredictNut":{__rrshift__:[5,2,1,""]},"nutsml.network.TrainValNut":{__rrshift__:[5,2,1,""]},"nutsml.plotter":{PlotLines:[5,1,1,""]},"nutsml.plotter.PlotLines":{__call__:[5,2,1,""],__init__:[5,2,1,""],reset:[5,2,1,""]},"nutsml.reader":{DplyToList:[5,1,1,""],ReadImage:[5,1,1,""],ReadLabelDirs:[5,1,1,""],ReadPandas:[5,1,1,""]},"nutsml.reader.DplyToList":{__call__:[5,2,1,""]},"nutsml.reader.ReadImage":{__call__:[5,2,1,""]},"nutsml.reader.ReadPandas":{__init__:[5,2,1,""],dply:[5,2,1,""],isnull:[5,4,1,""]},"nutsml.stratify":{CollectStratified:[5,1,1,""],Stratify:[5,1,1,""]},"nutsml.stratify.CollectStratified":{__rrshift__:[5,2,1,""]},"nutsml.stratify.Stratify":{__rrshift__:[5,2,1,""]},"nutsml.transformer":{AugmentImage:[5,1,1,""],ImageAnnotationToMask:[5,1,1,""],ImageChannelMean:[5,1,1,""],ImageMean:[5,1,1,""],ImagePatchesByAnnotation:[5,1,1,""],ImagePatchesByMask:[5,1,1,""],RandomImagePatches:[5,1,1,""],RegularImagePatches:[5,1,1,""],TransformImage:[5,1,1,""],map_transform:[5,3,1,""]},"nutsml.transformer.AugmentImage":{__init__:[5,2,1,""],__rrshift__:[5,2,1,""],by:[5,2,1,""]},"nutsml.transformer.ImageAnnotationToMask":{__rrshift__:[5,2,1,""]},"nutsml.transformer.ImageChannelMean":{__call__:[5,2,1,""],__init__:[5,2,1,""],train:[5,2,1,""]},"nutsml.transformer.ImageMean":{__call__:[5,2,1,""],__init__:[5,2,1,""],train:[5,2,1,""]},"nutsml.transformer.ImagePatchesByAnnotation":{__rrshift__:[5,2,1,""]},"nutsml.transformer.ImagePatchesByMask":{__rrshift__:[5,2,1,""]},"nutsml.transformer.RandomImagePatches":{__rrshift__:[5,2,1,""]},"nutsml.transformer.RegularImagePatches":{__rrshift__:[5,2,1,""]},"nutsml.transformer.TransformImage":{__call__:[5,2,1,""],__init__:[5,2,1,""],by:[5,2,1,""],register:[5,5,1,""],transformations:[5,6,1,""]},"nutsml.viewer":{PrintColType:[5,1,1,""],ViewImage:[5,1,1,""],ViewImageAnnotation:[5,1,1,""]},"nutsml.viewer.PrintColType":{__call__:[5,2,1,""],__init__:[5,2,1,""]},"nutsml.viewer.ViewImage":{__call__:[5,2,1,""],__init__:[5,2,1,""]},"nutsml.viewer.ViewImageAnnotation":{SHAPEPROP:[5,6,1,""],TEXTPROP:[5,6,1,""],__call__:[5,2,1,""],__init__:[5,2,1,""]},"nutsml.writer":{WriteImage:[5,1,1,""]},"nutsml.writer.WriteImage":{__call__:[5,2,1,""],__init__:[5,2,1,""]},nutsml:{batcher:[5,0,0,"-"],booster:[5,0,0,"-"],checkpoint:[5,0,0,"-"],common:[5,0,0,"-"],config:[5,0,0,"-"],datautil:[5,0,0,"-"],examples:[6,0,0,"-"],fileutil:[5,0,0,"-"],imageutil:[5,0,0,"-"],logger:[5,0,0,"-"],network:[5,0,0,"-"],plotter:[5,0,0,"-"],reader:[5,0,0,"-"],stratify:[5,0,0,"-"],transformer:[5,0,0,"-"],viewer:[5,0,0,"-"],writer:[5,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","staticmethod","Python static method"],"5":["py","classmethod","Python class method"],"6":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:staticmethod","5":"py:classmethod","6":"py:attribute"},terms:{"0x0000000009331048":5,"0x00000000093310d0":5,"0x0000000009331158":5,"0x00000000093311e0":5,"0x0000000009331268":5,"0x00000000093312f0":5,"0x0000000009331378":5,"0x0000000009331488":5,"0x0000000009331510":5,"0x0000000009331598":5,"0x0000000009331620":5,"0x00000000093316a8":5,"0x0000000009331730":5,"0x00000000093317b8":5,"0x0000000009331840":5,"0x00000000093318c8":5,"0x0000000009331950":5,"0x00000000093319d8":5,"0x0000000009331a60":5,"0x0000000009331ae8":5,"0xbf160e8":21,"10x20x3":5,"128x128":11,"128x64":23,"128x64x3":23,"1st":18,"1x10x20":5,"1x1x5x3":5,"1x3":5,"213x320":[5,17],"213x320x3":[5,17],"2x1x2x3":5,"2x1x5x3":5,"2x213x320x3":12,"2x2x3":5,"2x3":5,"2x3x2":5,"32x32":13,"3rd":18,"3x4":5,"5x4x3":5,"abstract":5,"boolean":[5,21],"case":[1,5,11,19,21,23],"class":[0,5,7,10,12,13,17,19,21,22,23,24],"default":[5,13,22,24],"final":[0,13,19,21],"float":[1,5,12,13,21],"function":[0,2,4,5,12,13,18,19,20,21,22,23,24],"import":[1,3,5,11,12,13,14,21,22],"int":[5,12,13,21],"new":[5,13,23],"return":[1,2,5,10,13,17,19,21,22,23],"short":[2,21],"static":5,"throw":13,"true":[5,12,13,19,21],"try":[12,21],"while":[5,12,13,21,22],Adding:21,And:5,But:21,For:[1,2,3,4,5,11,12,13,15,17,18,19,20,21,22,23,24],NOT:5,Not:13,One:5,The:[1,2,4,5,11,12,13,14,17,18,19,20,21,24],There:[13,21],These:[2,23],Use:[5,19,22],Used:5,Useful:5,Using:21,Will:5,__call__:[5,7],__future__:5,__init__:5,__rrshift__:5,aabbccdde:5,about:[5,22],abov:[5,11,12,13,19,21],abs:5,absolut:5,acc:[2,4,5,13,18,19,20],accept:13,access:[5,14],accord:5,accordingli:13,account:13,accur:[13,22],accuraci:[2,4,5,10,13,18,19,20,22],achiev:[5,19,21],across:[5,22],action:0,activ:[3,13,19],actual:[5,11,17,21],adam:[13,19],add:[1,5,12,13,19,21],add_channel:5,add_n:5,add_two:21,added:[5,13,23],adding:19,addit:[11,13,21,23,24],address:5,adjust:[5,11,23],advantag:[13,19,21],affinetransform:5,after:[5,11,12,13,19,21,22],again:[5,11,13,21,22],agg:1,aggreg:5,agre:23,aim:0,airplan:13,all:[2,5,13,19,21,22,23],allow:[5,12,13,17,18,20,21,24],almost:2,alon:11,alpha:5,alreadi:[5,21],also:[1,5,11,13,17,18,19,21,23,24],altern:[1,5,20,21],alwai:11,amount:[4,11,13],anaconda:3,analog:13,analysi:5,angl:[5,11],ani:[5,11,13,19],anno:5,annoarg:5,annocol:[5,24],annot:[5,10],annotation2coord:5,annotation2mask:5,annotation2pltpatch:5,anoth:[5,12,19],anymor:[5,13],anyth:[5,21],apart:[18,19],api:5,appear:[5,20],append:5,appli:[5,11,13,21,23],applic:[4,13],arang:5,arbitrari:12,arbitrarili:[21,23],architectur:[12,13,14],archiv:21,arg:[5,7,13],argmax:[13,19],argument:5,around:5,arr:5,arr_to_pil:5,arrai:[5,10,12,13,16,17,18],arrang:[2,4,5,21,24],artifici:22,arxiv:5,as_grei:5,aspect:5,assign:21,assum:[1,5,11,19,21],astyp:[5,23],attribut:5,audio:21,aug_contrast:11,aug_rot:11,augment1:5,augment2:5,augment:[2,4,5,10,12,16,19,23,24],augment_contrast:11,augment_flip:11,augment_imag:11,augment_img:5,augment_rot:11,augmentimag:[4,5,10,11,13],autoencod:[5,6,12],automat:[13,21],avail:[2,5,13,23],averag:[13,19,20],avoid:[0,22],awai:13,axes:[5,12,20],axi:[5,12,17,20,23],back:21,backend:1,backgroundcolor:5,bad:[5,12,22],balanc:22,bar:5,base:[2,4,5,7,9,10,17,19,21,22],basedir:5,basepath:5,basic:[16,18,20],bat:3,batch:[4,5,10,11,16,18,19,20,22],batch_siz:[4,13,19],batched_label:1,batched_pr:1,batcher:[0,4,12,13],batchlog:18,batchsiz:[5,12,13],becaus:21,becom:[5,12,21],been:[5,21],befor:[5,12,13,21,22],behavior:22,being:[20,21],belong:[5,22],below:[4,5,13,19],benchmark:13,best:5,beta:5,better:5,between:[5,10,11,13],beyond:[5,10],bia:22,bicub:5,bilinear:5,bin:3,binari:21,bit:[10,19],black:2,blanklin:5,blob:9,blur:[11,13],bmp:[1,5],book:21,bool:5,boost:[5,10],booster:0,both:[5,11,12,17,19],bottom:5,bracket:21,bright:[4,5,11,13],brighter:5,broken:21,build:[2,5,10,13,16,22],build_batch:[1,2,4,5,12,13,18,19,20,22],build_image_batch:[5,12],build_number_batch:5,build_one_hot_batch:5,build_pred_batch:[5,19],build_tensor_batch:[5,12],build_vector_batch:5,buildbatch:[0,4,5,10,11,12,13,19],built:5,call:[2,4,5,11,13,18,19,22,23],camelcas:[0,23],can:[2,3,4,5,11,12,13,14,16,18,19,21,22,23],cannot:12,canon:16,care:21,categori:13,categorical_accuraci:[4,13,19],categorical_crossentropi:[13,19],caus:5,center:5,centers_insid:5,cfg:[5,14],chain:[5,13,23],chanc:11,chang:[4,5,11,13,22],change_bright:5,change_color:5,change_contrast:5,change_sharp:5,changebright:23,channel:[5,10,12,17,23],channelfirst:[5,12],character:4,check:[4,5,16],checknan:[5,10],checkpoint103:5,checkpointnam:5,checkpointspath:5,chosen:11,cifar10:13,cifar10_cnn:13,cifar:[5,6,16],circl:5,class0:5,class1:5,class2:5,class_id:5,class_weight:1,classic:13,classif:[5,13,19,22],classifi:[13,16,21,22],classmethod:5,clean:0,clear:5,clear_fold:5,clearli:[4,21],click:10,clip:5,clockwis:5,clone:3,close:[5,18,20,21,22],cnn:13,cnn_classifi:[5,6,13],cnn_train:[5,6,13],code:[0,1,2,3,4,9,11,12,16,18,19,20,21,23],col1:5,col2:5,col:[5,12,18,21],col_map:5,collect:[0,1,3,4,5,11,13,19,21,22],collectstratifi:5,colnam:[5,18],color:[4,5,11,12,13,17,23,24],colum:12,column:[1,5,10,11,12,13,17,18,20,21,23,24],columnb:5,com:[3,5,9],combin:[5,11,12,21,22],come:[5,12,13],comma:21,common:[0,2,4,11,13,17,19,21,22,23],compar:13,compil:[5,13,19],complet:[4,5,11,13,16,19],complex:[12,13,18,20,23],compon:[4,21],compos:[12,17],composit:4,comput:[1,5,10,13,19,20],compute_metr:1,concept:16,conclud:[11,20,21],condit:23,confer:5,confid:[5,10],config:[0,1,14],configdict:5,configpath:5,configur:[5,16],confirm:17,consequ:[11,13,19],consider:22,consist:[5,13],constrain:5,constraint:[5,13,22],construct:[5,11,12,17,18,19,21],consum:[1,2,5,11,12,13,17,18,19,20,21,22,23,24],contact:5,contain:[5,10,11,12,13,17,18,21,22,23,24],content:[1,13,21],context:5,continu:[5,13],contrast:[4,5,11,13,23],contribut:2,control:5,conv_autoencod:[5,6],conveni:[17,21],converg:[5,20],convers:21,convert:[5,10,12,13,19,21,23],convert_label:12,convertlabel:[5,10,12,19,21],convolut:[5,13],convolution2d:[13,19],coordin:5,copi:[0,5,11,12,13],corner:5,correct:12,correspond:[5,11,12,13,20,21,23],cos:5,could:[2,13,19,20,21,22],count:[5,21],counter:5,countvalu:[1,5,21,22],cours:[13,18,21],cov:0,cover:0,coverag:0,cpu:[4,5,12],cran:5,creat:[0,1,3,5,12,13,14,16,18,20,21,22,23],create_filenam:5,create_fold:5,create_net:5,create_network:[4,5,7,8,9,13],create_temp_filepath:5,creation:[5,12],crop:[4,5,10],crop_cent:5,crop_squar:5,csv:[4,5,16,18,22],csvreader:0,current:[5,21],custom:[2,11,13,23],cycl:19,data:[0,1,2,4,5,9,10,11,12,14,16,17,19,23,24],databas:[4,21],datafram:5,datapath:5,dataset:13,datatyp:5,datautil:0,deactiv:3,deal:17,debug:[5,12,21],decod:[7,21],dedic:21,dedup:21,deep:[2,4,11,12,13],def:[2,5,13,21,23],defin:[4,5,11,13,18,21],degre:[5,11],delet:[5,18],delete_fil:5,delete_fold:5,delete_temp_data:5,delimit:5,deliv:21,demand:17,demonstr:[18,20,21],dens:[13,19],depend:[4,5,10,12,13,22],deprec:5,deriv:[5,11],describ:[0,4,5,11,13,19,21],descript:[2,5,20],design:[4,21,24],detai:5,detail:[2,4,5,10,12,13,15,20,21],determin:5,determinist:5,dev:5,develop:5,dict:5,dictionari:[5,14,21,22],diff:7,differ:[4,5,7,12,13,19,21,22],dim:5,dimens:[5,12,23],dir:5,direct:5,directli:[1,5,13,17,19,21,22],directori:[5,10,16],disabl:5,disk:23,displai:[5,10,11,13,17,20,24],distinguish:23,distort:5,distort_elast:5,distribut:[5,12,13,22],divid:[11,12,13],doc:5,doctest:[4,5],document:[2,5,21],dodger487:5,doe:[5,12,13,21,22],doesn:[0,5],dog:13,don:[3,5,13,19],done:22,doubl:11,down:[5,10,22],download:21,downrnd:5,downsampl:5,dply:5,dplydatafram:[5,10],dplyfram:5,dplyr:5,dplython:5,dplytolist:[5,10],draw:5,drop:[5,21],dropnan:5,dropout:13,dtype:[5,12,13,17,23],due:23,duplic:[5,21],dure:18,e_acc:[4,13,19],each:[4,5,11,12,13,20,21,22,23],earli:13,easi:[2,13,15,21,23],easiest:19,easili:[4,11,12,13,14,19,21,22,23],edgecolor:5,edu:21,effect:0,effici:[2,17],either:[5,12,13],elast:5,element:[0,5,7,17,21],ellips:5,els:[1,5,21],emit:19,empir:[5,10],empti:[5,21,22],enabl:[4,5,13,17],encod:[5,12,13,21],encount:5,encourag:5,end:[5,13,19,21,22],enhanc:5,enough:[5,13],ensur:[0,3,5,12,13,19,22],enter:3,entir:[5,13,17,21],enumer:5,epoch:[1,4,5,12,13,14,18,19,20,22],epochlog:18,equival:[18,21],error:[0,12],especi:13,estim:13,eval:1,evalnut:5,evalu:[1,4,5,16,19],even:[5,21,22],everi:[5,11,13,20,21,22],every_n:[5,20,21],every_sec:[5,20],exampl:[0,2,5,11,12,14,16,17,18,19,20,21,22,23,24],exce:5,except:[5,10,13,23],excerpt:11,exclud:5,exclus:5,execut:11,exist:[3,5],expect:[5,13,19,21,23],explain:2,explan:4,explicitli:[18,19],express:[4,5,21],ext:5,extend:[2,4,12,15,23],extens:[5,23],extract:[5,10,12,13,17,18,21,23],extract_patch:5,extrem:12,eye:[5,21],f1_score:5,facecolor:5,facilit:20,factor:5,factori:5,fail:5,fals:[5,13],familiar:2,faq:[0,2],far:13,fashion:[13,21,23],fchollet:9,feasibl:13,featur:[11,18,21],feed:19,figsiz:[5,13],figur:[5,20],file:[1,4,5,10,13,16,17,18,20],filenam:[5,13,17,21],filepaht:5,filepath:[5,13,18,20,21,22],filepattern:5,fileutil:0,fill:5,filter:[2,4,5,12,14,21],filtercol:21,filterfunc:5,find:5,fine:5,finish:20,first:[5,11,12,13,17,21,22],first_even:21,fit:13,fit_gener:19,fix:5,flag:[5,12,19],flat:5,flatten:[4,5,13],flattencol:1,flip:[4,5,11,12,13],fliplr:[4,5,11,13],flipud:[5,11],float32:[4,5,12,13],float64:5,floatimg2uint8:5,flow:[1,2,4,5,13,15,21],flower:22,fly:[21,24],fmt:5,fmtfunc:21,fold:22,folder:[5,13],follow:[0,1,2,3,4,5,11,13,14,17,18,19,20,21,22,23,24],foo:5,forc:[3,13],form:[5,13],format:[0,5,12,13,14,20,21,23],formatt:12,found:[4,13,16],four:21,fourth:21,fpath:5,fraction:5,fragment:11,frame:5,framework:12,freeli:2,frequenc:[5,22],frequent:14,from:[1,3,5,10,11,12,13,14,17,18,19,20,21,22,23],front:5,full:[5,17],func:5,further:[19,21],furthermor:[5,17,21],gamma:5,gave:21,gener:[0,4,5,11,12,13,19,20,21,22],geometr:[5,10],geometri:5,get:[1,5,11,13,20,21],getcol:5,getpatch:5,getsitepackag:3,gif:[1,5,12,17,23,24],git:[0,3],github:[3,5,9],give:[2,21],given:[5,11,13,17],glob:13,going:21,good:[5,12,13,21,22],gpu:[2,4,5,10,12],grai:[1,5,13,17],graph:5,graphic:1,gray2rgb:[5,12,23],grayscal:[5,12,17,23],grid:[5,10],group:5,group_bi:5,group_sampl:5,guid:23,had:21,halv:11,hand:[5,13],handl:12,hang:12,happi:19,has:[0,1,5,12,21,22],hasattr:1,hashabl:5,hat:5,have:[2,5,11,12,13,17,18,19,21,23],haven:13,hd5:[5,13],hdf5:13,head:[5,21],header:[5,21],height:[5,23],help:[11,12,13,20,23],here:[3,9,11,12,13,17,18,21,22,24],high:[5,10,13],higher:[5,11],highest:[13,19],histogram:5,hold:22,home:5,horizont:[5,11,13],hot:[5,12,13,21],how:[12,13,14,16,17,18,19,22],howev:[5,12,13,19,21,23],html:[0,5],http:[3,5,9,21],hyper:5,ics:21,idea:13,ident:[4,5,11,13,21],ids:[5,10],ifilt:4,ignor:[5,12],imag:[2,4,5,7,10,12,13,16,20,21,22],image_channel_mean:5,image_mean:5,image_patch:5,imageannotationtomask:[5,10],imagechannelmean:[5,10],imagecol:5,imageenh:5,imageid:5,imagemean:[5,10],imagenam:23,imagepatchesbyannot:[5,10],imagepatchesbymask:[5,10],imagepath:[5,11,13,17,24],images:5,imageutil:0,imarg:5,imbal:22,img0:13,img123:13,img19:13,img1:5,img2:5,img456:13,img789:13,img:[5,13],img_format:[5,11,12,17,23,24],img_sampl:5,imgcol:[5,24],immedi:[11,20],immut:0,implement:[2,4,5,13,21,23],impress:[2,20],imshow:5,in_ndarrai:12,inch:5,includ:[11,13],inclus:5,incomplet:4,incorrect:[5,10],increas:[4,11,13],inde:23,independ:[2,5,11,13],index:[1,2,5,12,13,19,20,24],indic:[5,12,13,19,20,21,24],individu:[5,12,20,21],infer:[12,19],infinit:19,info:5,inform:[5,10,12,13,14],ini:0,inifil:0,inpath:5,input:[0,4,5,11,12,13,19,22,23,24],input_shap:[13,19],insert:21,insid:5,inspect:[13,21],inspir:5,instal:[0,2],instanc:[0,1,2,4,5,11,13,17,18,19,21,22,23,24],instead:[5,12,13,17,20,21,23],integ:[5,10,12,13,21],integr:5,interest:20,intern:5,interpol:[5,13],interrupt:[5,13],interv:[5,13],introduc:[13,16,22,24],introduct:[0,2,5,21],invalid:5,invok:13,iri:[21,22],is_even:21,is_odd:[5,21],isjson:5,islic:[4,21],isloss:[5,13],isnan:5,isnul:5,item:[0,5,13,17,23],iter:[0,1,4,5,10,13,21],itertool:[4,5,21],its:[2,5,11,13,16,21],itself:[4,5],jpg:[1,5,11,13,17,23,24],json:[5,14],just:[0,5,11,17,20,21],kappa:5,keep:5,kei:[5,11],kept:5,kera:[5,9,10,13,18,19],kerasnetwork:[5,10,13,19],keyfunc:5,keyword:5,know:[3,19],knowledg:13,known:22,kwarg:[5,7],label2int:21,label:[1,2,5,10,12,13,16,17,19,22,23,24],labelcnt:5,labelcol:[5,22],labeldir:5,labeldist:[5,22],lambda:[4,5,13,21,22],larg:[4,11,13,17,20,21,22],larger:[5,12,21,23],largest:[5,17],lasagn:[5,10,13],lasagnenetwork:[5,10],lasgan:5,last:[5,12,13,16],later:[13,22,23],latest:[1,5],layer1:14,layer2:14,layer:[5,9],layout:[5,13,20,24],lazi:21,lazili:[17,21],lead:4,learn:[2,4,5,11,12,13,17,19,21,22],left:5,len:[12,21,22],length:[5,12],let:[11,12,13,19,20,21,22],lib:3,librari:[2,12,13,21],like:[0,5,19,21],line:[1,5,10,13,21],linear:5,linewidth:5,link:5,linux:3,list:[0,4,5,10,12,13,17,18,19,21,22,23],live:20,load:[4,5,10,14,16,19,21,22,24],load_config:5,load_data:13,load_imag:5,load_nam:8,load_sampl:[4,7,8,9,13],load_weight:5,loader:4,locat:[5,10,21],log:[4,5,10,16],log_batch:18,log_epoch:18,logcol:5,logger:[0,4],logtofil:[5,10,18],look:[2,12,13,21],loop:[18,19,22],loss:[2,4,5,10,13,18,19,20],lower:5,lumin:5,mac:3,machin:[4,5,21,22],maet3608:3,maet:[0,3,5],maetschk:5,main:17,make:[0,5,23],makefirst:5,manag:5,mani:[4,5,13,17,20,21],manner:5,manual:18,map:[5,13,19,21],map_transform:5,mapcol:21,mask:[5,10,11,17,24],mask_choic:5,mask_patch:5,mask_wher:5,maskcol:5,master:9,match:[5,18,19,23],matplotlib:[1,5],matplotlibrc:1,matrix:21,max:5,maximum:5,maxpooling2d:13,mayb:21,mean:[1,4,5,10,13,18,19,20,21,22],meaning:21,measur:13,medic:22,memori:[5,13,17,21,22],messag:[1,12],meta:17,method:[5,13,19,21,22],metric:[5,13,18,19],might:5,mini:[4,5,11,12,13,18,19],minim:[5,10],minimum:5,miss:[0,5],mix:4,mixtur:12,mixup:[5,10],mlp_classifi:[5,6],mlp_train:[5,6],mlp_view_misclassifi:[5,6],mmap_mod:21,mnist:[5,6,7,13],mnist_cnn:9,mnist_mlp:9,mode:[4,5,14],model:[5,13,19],modifi:[2,4,5,9,13],modul:2,modular:21,momentum:5,mono:[23,24],monochrom:[5,11,23,24],monolith:4,more:[4,5,10,11,12,13,17,18,19,20,21,22,23],most:[5,12,21,23],move:[5,12,17],much:[12,22],multi:9,multipl:[1,5,12,17,20,21,23,24],multipli:11,must:[5,18],mutat:0,mxn:5,mxnx1:5,mxnx3:5,mxnx4:5,mxnxc:5,my_bright:[13,23],my_project:3,my_python_path:3,mynetwork:5,n_class:12,name:[0,5,10,13,17,18,21,23],namefunc:5,nan:[5,10],ndarrai:[5,13,17,23],nearest:5,necessari:[4,5],need:[0,1,4,5,11,12,13,17,19,20,21,22,23],neg1:5,neg2:5,neg:[3,5],neglect:21,nest:4,network:[0,1,2,4,10,12,14,16,18,20,22],neural:[4,5,13],new_imag:23,new_max:5,new_min:5,newlin:21,next:[11,12,13,17,18,20,21,22,23],nice:[5,13,21],nneg:5,no_alpha:5,non:5,none:[5,12,13,17,20,23],normal:[5,23],normalize_histo:5,notabl:13,notblack:2,note:[5,11,12,13,17,18,19,20,21,22],noth:[5,21],notic:21,now:[11,13,21,22,23],npatch:5,npo:5,npy:[1,5],npz:5,num_class:[4,5,13,19],num_epoch:[4,13],number:[0,5,11,12,13,18,19,20,21,22],numer:[5,21],numpi:[1,5,10,12,13,16,17,18],nut:[0,3,4,5,6,7,8,9,10,11,13,14,16,17,18,19,20,21,23,24],nut_:[17,23,24],nut_color:[5,11,12,17,23,24],nut_filt:2,nut_funct:[21,23],nut_grayscal:[5,12,17],nut_monochrom:[11,12,23,24],nutfunct:[5,7],nutmsml:5,nutsflow:[1,3,5,7,21,22],nutsink:5,nutsml:[0,2,3,4,13,14,21],nutsourc:5,nx2:5,object:[5,13,21,22],obvious:22,occasion:22,occlud:5,occlus:5,occur:[5,11,22],odd:[21,22],often:[4,12,13,18,19,20,22,24],old_max:5,old_min:5,omit:13,onc:[19,20,21],one:[5,11,12,13,20,21],one_hot:[4,5,12,13],onehot:[5,21],ones:[5,21],onli:[5,11,12,13,20,21,22],onlin:5,open:[3,5,18,21],oper:[4,5,21,22,23],opt:5,optim:[5,13,19],order:[5,16,22],ordereddict:5,org:5,organ:[4,5,13,16,21],origin:[5,7,11,13,22],other:[4,5,10,13,14,18,20,21],otherwis:[0,5,17],our:[11,12,14],ourselv:23,out:[2,12,19,21],out_lay:5,out_ndarrai:12,output:[0,1,4,5,11,12,13,18,19,23,24],outsid:18,over:[0,2,5,10,13,19,20,21,22],overal:2,overlai:24,overview:[2,11,12],own:[2,15],packag:[2,3],pad:13,page:2,pair:17,panda:[4,5,10],pandas_t:5,paper:5,parallel:12,paramet:[5,11,13,14,20,21,24],paramspath:5,pariti:22,parser:5,part:[11,13,17,19],partial:[13,22],partit:[5,10],partitionbycol:[5,10],pass:[0,5,11,13],patch:[5,10],patch_it:5,patches_api:5,path:[5,10,13,17,21],pathfunc:5,patient:22,pattern:5,paus:[5,11,13,24],pep:0,per:[5,10,13,18,19,20,24],perceptron:9,perform:[4,5,11,13,19,22,23],period:5,perturb:13,phase:[5,7,12],pic:20,pick:[5,13],pickl:5,piec:[11,13],pil:5,pil_img:5,pil_to_arr:5,pillow:5,pip:3,pipelin:[6,8,9,12,13,16,18,19,21,22],pixel:[5,13],plain:[1,23],platform:0,pleas:[0,13],plot:[4,5,10,16,18],plot_batch:20,plot_epoch:20,plot_ev:13,plot_squar:20,plotarg:5,plotlin:[5,10,13,20],plotter:0,plug:[2,12,19],pluggi:0,plugin:0,png:[1,5,13,17,20,23,24],point:[4,5,16],polylin:5,polyline2coord:5,pool_siz:13,pos1:5,pos2:5,pos3:5,pos:[3,5],posit:[5,13],possibl:[5,12,17],power:21,practic:[5,13],pre:[2,4,17],preced:[5,22],precis:5,pred:5,pred_batch:13,pred_fn:5,predcol:5,predict:[5,7,12,16,19],predictnut:5,prefer:[0,21],prefetch:[5,12],prefix:[5,17],preprocess:[9,19,22],prerequisit:13,preserv:5,press:[5,11],previou:19,principl:21,print:[2,3,4,5,10,12,13,17,18,19,21,22,24],print_funct:5,print_lay:5,printcoltyp:[5,10,13,17,23],printout:17,printprogress:[4,13],prob:5,probabl:[0,1,5,11,13,19],problem:[13,19,21],problemat:12,proc:5,process:[2,4,5,10,12,13,17,21],processor:0,produc:[5,12,18],product:5,progress:[4,11],project:0,properli:21,properti:5,prove:5,provid:[0,2,4,5,11,12,13,14,16,17,18,19,20,21,22,23,24],pseudo:22,pshape:5,pull:[19,21],purpos:[11,19,24],push_doc:0,put:21,py3k:21,pypi:3,pyplot:5,pyplot_api:5,pytest:[0,3],python:[0,3,4,5,13,21,22,23],quadrat:20,qualiti:5,quick:2,rais:[5,10],rand:[5,22],random:[4,5,10,11,13,22],random_downsampl:5,randomimagepatch:[5,10],randomli:[5,10,11,13,22],rang:[0,4,5,11,12,13,17,18,19,20,22,23],rare:23,rate:5,ratio:[5,12,22],rbg:5,reach:13,read:[0,2,4,5,10,11,12,16,17,18,20,23,24],read_imag:[2,5,6,11,12,13,19,23],read_sampl:12,readabl:[4,12,13,21],readcsv:[0,21],reader:[0,4,21],readi:21,readimag:[1,5,10,11,12,13,17,23,24],reading_sampl:0,readlabeldir:[5,10,13,21],readlin:21,readpanda:[5,10,21,22],readthedoc:5,real:22,realist:22,rearrang:13,reason:[13,17],recent:5,recognit:[5,13],recommend:2,record:[21,22],recreat:5,rect:5,rectangl:5,rectangular:5,reduc:5,refactor:21,refer:5,regard:5,region:5,regist:[5,11,13,23],regular:[5,10,13],regularimagepatch:[5,10],rel:5,relat:5,relev:5,reli:21,reload:21,relu:13,remain:[4,5,21],remaind:13,remov:[5,12,21],renam:21,reopen:21,reorder:[12,21],repeat:22,replac:[5,13,17,21],replacenan:5,report:[13,19],repres:[5,13,21],represent:5,request:[5,21],requir:[11,12,13,19,21,22],rerang:[4,5,13,23],rerun:[21,22],rescal:5,reset:5,reshap:5,resiz:[4,5,11,13,23],respect:5,restart:21,result:[1,5,11,12,21,22,23],retlabel:5,retriev:[13,19],retvalu:13,reus:5,reusabl:2,rgb2grai:5,rgb:[1,5,12,13,23],rgb_arr:5,rgba:[1,5,13],right:[5,12],risk:[5,10],rnd:5,robust:4,rootdir:0,rot:5,rotat:[4,5,10,11,13],roughli:[11,16,22],round:5,routin:5,row:[5,12,20,21,23,24],rst:0,rstudio:5,run:[0,1,3,5,7,12,13,19,20,21,23],runner:[5,6],runtimeerror:5,same:[0,4,5,11,12,13,17,18,21,22,24],same_lett:5,same_par:22,sampl:[1,2,3,4,5,10,11,12,13,16,17,18,19,20,22,23,24],sample_labeled_patch_cent:5,sample_mask:5,sample_patch_cent:5,sample_pn_patch:5,save:[5,13,14],save_best:[5,13],save_imag:5,save_weight:5,scala:21,scale:[1,4,5,13,17],scatter:[5,22],scenario:12,schemat:12,scikit:[5,17],score:5,screen:5,script:3,search:[2,5],sec:5,second:[0,1,5,11,12,13,17,20,22,24],section:[11,12,13,16,17,18,19,20,22,23,24],see:[0,1,4,5,9,10,11,12,13,15,22,23],seed:[5,22],seen:13,segment:24,select:[5,10,22],self:5,separ:21,sequenc:[5,13,22],sequenti:[5,13,19],server:21,session:[0,5],set:[4,5,11,12,13,17,19,20,21,22,23],set_default_ord:5,setosa:[21,22],setup:3,sgd:5,shape:[1,5,12,13,17,19,23],shapeprop:5,shapestr:5,share:4,sharp:5,shear:[4,5,13],shear_factor:5,shell:3,shift:5,ship:13,shorten:21,shorter:12,should:[0,5,11,22],show:[4,5,7,12,13,14,17,18,20,23,24],show_imag:13,showcas:21,shown:[5,13],shuffl:[2,4,5,12,13,19,22],shuffle_sublist:5,side:0,simard:5,similar:[11,18,19],similarili:21,similarli:[12,17],simpl:[12,17,20,21],simpler:[17,21],simpli:[5,18,21],simplifi:[5,13,14],sin:5,sinc:[5,12,13,20,21,22],singl:[5,11,12,13,19,24],sink:[0,13,19,21,22],site:3,size:[4,5,12,13,17,22],sketch:[18,19,20],skim:2,skimag:5,skip:[0,4,5,19,21],skip_empti:21,skip_head:21,skiphead:21,slightli:[12,13],small:13,smaller:5,smallest:[5,13,17],smooth:5,softmax:[5,13,19],softwar:3,some:[2,4,5,12,13,19,23],someth:5,sometim:[5,11,12,17,24],sort:5,sourc:[0,3,4,5,7,8,9,21],space:21,spec:5,special:5,specif:[4,5,13,21,22],specifi:[5,11,12,13,17,18,19,20,23,24],speed:[5,20],sphinx:0,spline36:13,split:[4,5,10,13,16,21],splitrandom:[5,10,12,13,22],splitter:4,squar:[0,5],ssss:0,stabl:5,stablerandom:[5,22],stack:[5,12,13],standard:[1,13],start:[0,2,5,12,21,22],state:[5,24],statement:13,stefan:5,stem:21,step:[4,5,11],still:[5,17,21],stmt:0,stop:[5,13],store:[4,5,13,14,21],str:[5,17,21],strategi:[11,13,17],stratif:[5,22],stratifi:[0,2,3,4,10,16],stream:[13,19],stride:[5,14],string:[5,10,12,17,21],strip:21,structur:[5,12,21],style:23,sub:[5,13],sub_mean:5,sublist:[5,12],submodul:6,subsampl:5,subsequ:11,subset:20,subtract:[5,10],succinctli:21,suffici:5,suitabl:[21,23],sum:[2,5],support:[1,5,11,12,19,21,22],sure:23,surnam:5,symmetr:5,syntact:18,syntax:[5,21],system:[5,10,13,14,21],t_acc:[4,13,18,19,20],t_loss:[1,4,13,18,19,20],tab:21,tabl:[4,5,10],take:[4,5,11,13,17,19,21,22,24],target:[5,12,19],targetcol:5,task:[4,16,22,23,24],tast:2,tech:13,temp:5,temp_fold:5,temp_logfil:[5,18],temp_plott:5,templat:22,temporari:5,tensor:[5,12],test0:5,test11:5,test1:5,test:[4,5,10,11,12,13,14,17,18,19,21,22,23,24],test_:5,test_batch:0,test_boost:0,test_common:0,test_config:0,test_datautil:0,test_fileutil:0,test_imageutil:0,test_logg:0,test_network:0,test_read:0,test_sampl:[4,12,13,19],test_stratifi:0,test_transform:0,test_view:0,test_writ:0,text0:21,text11:21,text1:21,text:[4,5,21,24],textprop:5,than:[5,11,13,21],theano:5,thei:5,them:[5,11,12,13,21,22,23],themselv:13,therefor:[13,20,21,22],thi:[1,4,5,9,11,12,13,14,16,17,18,19,20,21,22,23,24],thing:12,third:5,those:[11,19,20,21],though:23,three:[12,13,18,21,22],threshold:2,through:[4,11,13],tif:[1,5],time:[5,11,17,20,22,24],tini:21,titl:[5,20],to_column:21,to_float:21,to_int:21,to_rgb:12,to_upp:21,togeth:[11,13],toi:[12,22],toint:21,tolist:5,too:[13,17],top:5,topic:[12,18,23],total:0,traceback:5,train:[1,2,4,5,7,8,9,10,11,12,14,16,18,20,21,22,23],train_err:5,train_fn:5,train_loss:5,train_network:5,train_on_batch:19,train_sampl:[2,4,12,13,18,19,20],trainvalnut:5,transform:[0,2,4,10,11,12,16,19,24],transformimag:[4,5,10,11,12,13,23],translat:[5,10],transpos:5,transspec:5,treat:11,tricki:12,trivial:21,tsv:[5,21],tupl:[0,5,13,17,18,20,21,22,23],tutori:[0,2,4,13,22],two:[5,11,12,13,20,21,23,24],txt:[5,16],type:[0,4,5,10,12,13,17,21,23],typeerror:1,typic:[4,5,11,12,13,19,22],uci:21,ugli:5,uint16:5,uint8:[4,5,12,13,17,23],unalt:[5,11],unbalanc:22,unchain:5,unchang:[5,13],under:[1,4,5,13,22],underli:5,understand:21,uniformli:[11,13],uniqu:5,unit:[2,22],unstructur:4,unwant:5,unzip:[4,5,13,18,19,20],updat:[0,11,14,20],upfront:[5,22],upgrad:1,upper:[5,21],upsampl:5,url:21,urllib:21,urlopen:21,usag:[5,16,20,21],use:[5,9,11,12,13,17,18,19,21,22,23],used:[1,5,11,12,13,18,21,23],useful:[5,11,13,17,18,21],user:5,uses:[5,13,22],using:[5,7,10,11,13,17,19,21,23,24],usual:[5,13,21,22],utf:21,util:5,v_acc:[13,19],v_loss:[13,19],val:[5,13,19,22],val_err:5,val_fn:5,val_loss:5,val_sampl:[13,19],valid:[4,5,10,16,19,22],validate_network:5,valu:[5,10,11,13,17,18,19,20,21],valueerror:5,vari:5,variabl:12,variou:5,vector:[5,12,13,19,21],verbos:[5,12,19],veri:[2,19,21,22],verif:0,verifi:[13,21],versa:[5,21],versicolor:[21,22],version:[1,3,5,13,17,22],vertic:[5,11],via:[3,5,11,13,17,18,20,21,23,24],vice:[5,21],video:4,view:[5,7,10,16],view_augmented_imag:[5,6],view_data:[5,6],view_imag:11,view_train_imag:[5,6],viewer:[0,24],viewimag:[5,10,11,13,24],viewimageannot:[5,10,13,24],vignett:5,violat:22,virginica:[21,22],virtualenv:3,visual:5,vnut:3,vstack:5,wai:[13,19],wait:5,want:[3,5,11,13,14,19,20,21,22,24],warp:5,web:[5,16],weight:[4,5,13],weights_cifar10:13,weights_keras_net:5,weights_lasagne_net:5,weightsfil:13,weightspath:5,welcom:0,well:[11,22],were:[5,13,19],what:[13,21,22],whatev:5,when:[1,4,5,12,13,19,20,21,23],where:[1,3,5,11,12,13,17,21],which:[4,5,11,12,13,19,21,23],white:21,wide:23,width:[5,23],wildcard:5,win32:0,window:[3,5,10,11,24],wise:[5,20],with_dtyp:5,within:[5,11,12,13,17,18],without:[0,13,19,21],word:[5,21],word_count:21,work:[3,5,12,13],world:22,would:[11,13,19,21,22],wrap:[4,5,10,12,13,19,21],wrapper:[5,10,13,19],write:[5,10,16,21],write_imag:[5,6,13],writeimag:[5,10,13],writer:0,written:[4,5,20],x213x320x3:12,x32x3:13,x3x213x320:12,x_test:13,x_train:13,xcol:5,xlsx:5,xrang:[1,4,5,13],y_test:13,y_train:13,yaml:[5,14],yco:5,ycol:5,yes:21,yet:5,you:[0,1,2,3,11,12,13,21,22,23],your:[1,2,3,15],ysin:5,zero:5,zip:[5,13]},titles:["Contributions","FAQ","Welcome to nuts-ml","Installation","Introduction","nutsml package","nutsml.examples package","nutsml.examples.autoencoder package","nutsml.examples.cifar package","nutsml.examples.mnist package","Overview","Augmenting images","Building Batches","CIFAR-10 Example","Configuration files","Custom nuts","Tutorial","Loading images","Logging data","Training networks","Plotting data","Reading data samples","Splitting and stratifying","Transforming images","Viewing Images"],titleterms:{"class":1,"default":1,arrai:[1,21],augment:[11,13],autoencod:7,basic:21,batch:[1,12,13],batcher:5,bleed:3,booster:5,build:12,can:1,canon:4,check:13,checkpoint:5,cifar:[8,13],cnn_classifi:[8,9],cnn_train:[8,9],code:13,common:5,config:5,configur:14,content:[5,6,7,8,9],contribut:0,conv_autoencod:7,convert:1,csv:21,custom:15,data:[13,18,20,21,22],datautil:5,directori:21,document:0,edg:3,environ:[0,3],error:1,evalu:13,exampl:[4,6,7,8,9,13],faq:1,file:[14,21],fileutil:5,flatten:1,format:1,guid:0,how:1,imag:[1,11,17,23,24],imageutil:5,imbalanc:1,importerror:1,indic:2,instal:3,introduct:4,kera:1,label:21,length:1,librari:4,load:[13,17],log:18,logger:5,mlp_classifi:9,mlp_train:9,mlp_view_misclassifi:9,mnist:9,modul:[1,5,6,7,8,9],name:1,network:[5,13,19],numpi:21,nut:[1,2,15],nutsml:[5,6,7,8,9],onli:1,overview:10,packag:[5,6,7,8,9],pipelin:4,plot:20,plotter:5,point:13,predict:[1,13],python:1,read:[1,13,21],read_imag:[8,9],reader:5,represent:1,result:13,runner:7,sampl:21,scalar:1,split:22,standard:3,stratifi:[5,22],style:0,submodul:[5,7,8,9],subpackag:[5,6],task:13,test:0,tkinter:1,train:[13,19],transform:[5,13,23],tutori:16,txt:21,unit:0,upgrad:3,use:1,valid:13,verif:3,view:24,view_augmented_imag:8,view_data:8,view_train_imag:[8,9],viewer:5,virtual:3,web:21,weight:1,welcom:2,what:1,write:13,write_imag:[8,9],writer:5}})