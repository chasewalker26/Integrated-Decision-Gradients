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
The location of the IDG source code is in util/attribution_methods/saliencyMethods.py

Example Usage
---
Inside of IDG/ is an example notebook example.ipynb
 * It showcases the use of IDG, as well as provides a visual comparison with IG, LIG, GIG, and AGI.
 * The img/ folder provides four correctly classified ImageNet images for experimentation

Replicating the Paper Results
---
The test results of the main paper are replicated with IDG/evaluations/evalOnImageNet.py.

#### evalOnImageNet.py Usage:
  * Run the script from IDG/evaluations/
  * Results will be saved in a folder called "test_results" in the IDG directory.
     * The results of each test (attribution and scoring method) will be saved as a appropriately named csv file. 

* To test IG with a pytorch pre-trained ResNet101 model on 2012 ImageNet validation data, use the following command:
  * `python3 evalOnImageNet.py --function IG  --image_count 5000 --model R101 --imagenet <DIR_TO_FOLDER>`
  * Where `<DIR_TO_FOLDER>` points to the 2012 ImageNet 50k validation set, and files in the folder should of the structure: "ILSVRC2012_val_00000001.JPEG"

* To perform the comprehensive test suite seen in the paper, the following command would be used:
    * `python3 evalOnImageNet.py --function <attr_method> --image_count 5000 --model <model> --imagenet <DIR_TO_FOLDER>`
    * Where `<attr_method> = {IG, LIG, GIG, AGI, IDG}` and `<model> = {R101, R152, RNXT}`
    * For a total of 15 script runs
