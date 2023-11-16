# Integrated Decision Gradients (IDG) PyTorch
The PyTorch implementation of the paper: 
 * Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision - [ArXiv Link](https://arxiv.org/abs/2305.20052v1)

**To cite this software please use:**

    @article{walker2023integrated,
      title={Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision},
      author={Walker, Chase and Jha, Sumit and Chen, Kenny and Ewetz, Rickard},
      journal={arXiv preprint arXiv:2305.20052},
      year={2023}
    }


Source Code
---
The location of the IDG source code is in AttributionMethods.py

Example Usage
---
This repository provides an example notebook example.ipynb
 * It showcases the use of IDG, as well as provides a visual comparison with IG, LIG, GIG, and AGI.
 * The img/ folder provides four correctly classified ImageNet images for experimentation

Replicating the Paper Results
---
The test results of the paper, and the comparisons shown in the supplementary results can be replicated with the testScript.py and allAttrComp.py files.

#### testScript.py Usage:
  * This was used to generate all quantitative results for the paper. 
  * Results will be saved in a folder called "tests".
     * Each test (attribution and scoring method) will be saved as an image of a graph, and a csv file so custom graphs can be generated. 

* To test IG with a pre-trained ResNet101 model on 2012 ImageNet validation data, use the following command:
  * `python3 testScript.py --function IG  --image_count 5000 --model R101 --imagenet <DIR_TO_FOLDER>`
  * Where `<DIR_TO_FOLDER>` points to the 2012 ImageNet 50k validation set, and files in the folder should of the structure: "ILSVRC2012_val_00000001.JPEG"

* To perform the comprehensive test suite seen in the paper, the following command would be used:
    * `python3 testScript.py --function <attr_method> --image_count 5000 --model <model> --imagenet <DIR_TO_FOLDER>`
    * Where `<attr_method> = {IG, LIG, GIG, AGI, IDG}` and `<model> = {R101, R152, RNXT}`
    * For a total of 15 script runs

#### allAttrComp.py Usage:
* Generates a PDF comparing the attribution methods as seen in the supplementary material and the qualitative results.
* To generate 50 comparisons using ResNet101 use this script as:
   * `python3 allAttrComp.py --image_count 50 R101 --model --imagenet <DIR_TO_FOLDER> --file_name <FILE_TO_SAVE>`
   * Where `<FILE_TO_SAVE>` is the location and name of the output PDF
