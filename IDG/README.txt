There are two main scripts in this code folder.


First is the testScript.py script. This was used to generate all quantitative results for the paper. Results will be saved in a folder called "tests". 
Each test (attribution and scoring method) will be saved as an image of a graph, and a csv file so custom graphs can be generated. 
    To test IG as done in the paper with this script use:
        python3 testScript.py --function IG --image_count 5000 --model 0 --imagenet <DIR_TO_FOLDER>

To perform all of the tests seen in the paper, the following commands would be used:
    R101
        python3 testScript.py --function IG  --image_count 5000 --model R101 --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function LIG --image_count 5000 --model R101 --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function GIG --image_count 5000 --model R101 --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function AGI --image_count 5000 --model R101 --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function IDG --image_count 5000 --model R101 --imagenet <DIR_TO_FOLDER>
    R152
        python3 testScript.py --function IG  --image_count 5000 --model R152 --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function LIG --image_count 5000 --model R152 --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function GIG --image_count 5000 --model R152 --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function AGI --image_count 5000 --model R152 --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function IDG --image_count 5000 --model R152 --imagenet <DIR_TO_FOLDER>
    ResNeXt
        python3 testScript.py --function IG  --image_count 5000 --model RESNXT --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function LIG --image_count 5000 --model RESNXT --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function GIG --image_count 5000 --model RESNXT --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function AGI --image_count 5000 --model RESNXT --imagenet <DIR_TO_FOLDER>
        python3 testScript.py --function IDG --image_count 5000 --model RESNXT --imagenet <DIR_TO_FOLDER>


The second script allAttrComp.py generates a PDF comparing the attribution methods as seen in the supplementary material and the qualitative results.
    To generate 50 comparisons using ResNet101 use this script as:
        python3 allAttrComp.py --image_count 50 R101 --model --imagenet <DIR_TO_FOLDER> --file_name <FILE_TO_SAVE>


In the directory which you point the scripts to for the imagenet images, the images in the folder should be of the name: "ILSVRC2012_val_00000001.JPEG"


Additionally, example.ipynb provides a notebook that shows the five attributions under comparison. The img/ folder holds four example imagenet images to test with.


Author response tests

python3 testScript.py --function IDG --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function IG --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function GBP --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function DL --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function IDGI --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function IGU --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function SG --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function GS --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function GC --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0
python3 testScript.py --function LPI --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 0

python3 testScriptPatternNet.py --function LPI --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function IDG --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function IG --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function GBP --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function DL --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function IDGI --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function IGU --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function GC --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function GS --image_count 1000 --model R101 --cuda_num 0
python3 testScriptPatternNet.py --function SG --image_count 1000 --model R101 --cuda_num 0

Adaptive sampling tests 

python3 testScriptAS.py --function IG --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 1
python3 testScriptAS.py --function IG_M --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 1
python3 testScriptAS.py --function IDG --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 2
python3 testScriptAS.py --function IDG_M --image_count 1000 --model R101 --imagenet ../../imagenet --cuda_num 2
