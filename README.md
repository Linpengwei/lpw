# ViT_Bamboo_NeurIPS_Code
- [Yuelu Shuyuan Cang Qin Jian (Si--Qi) Wenzi Bian](https://1drv.ms/u/c/4851c5065d666952/Ee3G4NuFwYhPpJG7W6gp4dMBzOSqif_FImlpor4MqKae-g?e=qLu1ws) - Data sources for the first stage of constructing the dataset and put the downloaded file into ViT_Bamboo_NeurIPS_Code/DATA. Use this datasets:
    ```bash
    python notebook/crop_images.py
    ```

- Test the topk result of the model:
    ```bash
    unzip DATA/SDD.zip
    sh test.sh
    ```