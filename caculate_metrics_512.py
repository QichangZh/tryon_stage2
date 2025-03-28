from metrics import FID, LPIPS, Reconstruction_Metrics, preprocess_path_for_deform_task, KID
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid = FID()
lpips_obj = LPIPS()
rec = Reconstruction_Metrics()
kid = KID() 

real_path = './datasets/deepfashing/train_lst_512_png' 
gt_path = '/datasets/deepfashing/test_lst_512_png'


distorated_path =  './PCDMs_Results/stage3_512_results'  
results_save_path =  distorated_path + '_results.txt'    # save path


gt_list, distorated_list = preprocess_path_for_deform_task(gt_path, distorated_path)
print(len(gt_list), len(distorated_list))

FID = fid.calculate_from_disk(distorated_path, real_path, img_size=(512,512))
LPIPS = lpips_obj.calculate_from_disk(distorated_list, gt_list, img_size=(512,512), sort=False)
REC = rec.calculate_from_disk(distorated_list, gt_list, distorated_path,  img_size=(512,512), sort=False, debug=False)
KID = kid.calculate_from_disk(distorated_path, real_path, img_size=(352,512))

print ("FID: "+str(FID)+"\nLPIPS: "+str(LPIPS)+"\nSSIM: "+str(REC)+"\nKID: "+str(KID))
with open(results_save_path, 'a') as ff:
    ff.write("\nFID: "+str(FID)+"\nLPIPS: "+str(LPIPS)+"\nSSIM: "+str(REC)+"\nKID: "+str(KID))