clear all; close all; clc

%% no transducer
path_to_data = '/Users/jacekdmochowski/PROJECTS/fus/data/SUBJECTS/JD_FUS_meat/SESSIONS/1/ACQUISITIONS/rfMRI_NO_TRANSDUCER/FILES'
filename = '1_3_12_2_1107_5_2_43_166037_202312111654092396031583_0_0_0.nii.gz'

nii_nt = load_untouch_nii(fullfile(path_to_data,filename));
bold_nt = double(nii_nt.img);

figure; 
subplot(221); imagesc(squeeze(mean(bold_nt(:,:,10,:),4))); title('no transducer','FontWeight','normal'); colorbar
subplot(222); imagesc(squeeze(mean(bold_nt(:,:,30,:),4))); colorbar
subplot(223); imagesc(squeeze(mean(bold_nt(:,:,50,:),4))); colorbar
subplot(224); imagesc(squeeze(mean(bold_nt(:,:,70,:),4))); colorbar

jdprint('meat_phantom_bold_no_transducer.png')

%% control
path_to_data = '/Users/jacekdmochowski/PROJECTS/fus/data/SUBJECTS/JD_FUS_meat/SESSIONS/1/ACQUISITIONS/rfMRI_control/FILES'
filename = '1_3_12_2_1107_5_2_43_166037_2023121116445648692001054_0_0_0.nii.gz'
nii_c = load_untouch_nii(fullfile(path_to_data,filename));
bold_c = double(nii_c.img);

figure; 
subplot(221); imagesc(squeeze(mean(bold_c(:,:,10,:),4))); title('control','FontWeight','normal'); colorbar
subplot(222); imagesc(squeeze(mean(bold_c(:,:,30,:),4))); colorbar
subplot(223); imagesc(squeeze(mean(bold_c(:,:,50,:),4))); colorbar
subplot(224); imagesc(squeeze(mean(bold_c(:,:,70,:),4))); colorbar

jdprint('meat_phantom_bold_control.png')

%% 100 kPa _1
path_to_data = '/Users/jacekdmochowski/PROJECTS/fus/data/SUBJECTS/JD_FUS_meat/SESSIONS/1/ACQUISITIONS/rfMRI_100kPa_1/FILES'
filename = '1_3_12_2_1107_5_2_43_166037_2023121116492210018119614_0_0_0.nii.gz'

nii_1_1 = load_untouch_nii(fullfile(path_to_data,filename));
bold_1_1 = double(nii_1_1.img);

figure; 
subplot(221); imagesc(squeeze(mean(bold_1_1(:,:,10,:),4))); title('100 kPa (scan 2)','FontWeight','normal')
subplot(222); imagesc(squeeze(mean(bold_1_1(:,:,30,:),4)))
subplot(223); imagesc(squeeze(mean(bold_1_1(:,:,50,:),4)))
subplot(224); imagesc(squeeze(mean(bold_1_1(:,:,70,:),4)))

%% 200 kPa _1
path_to_data = '/Users/jacekdmochowski/PROJECTS/fus/data/SUBJECTS/JD_FUS_meat/SESSIONS/1/ACQUISITIONS/rfMRI_200kPa_1/FILES'
filename = '1_3_12_2_1107_5_2_43_166037_2023121116495663667523326_0_0_0.nii.gz'

nii_2_1 = load_untouch_nii(fullfile(path_to_data,filename));
bold_2_1 = double(nii_2_1.img);

figure; 
subplot(221); imagesc(squeeze(mean(bold_2_1(:,:,10,:),4))); title('200 kPa (scan 2)','FontWeight','normal')
subplot(222); imagesc(squeeze(mean(bold_2_1(:,:,30,:),4)))
subplot(223); imagesc(squeeze(mean(bold_2_1(:,:,50,:),4)))
subplot(224); imagesc(squeeze(mean(bold_2_1(:,:,70,:),4)))

%% 300 kPa 
path_to_data = '/Users/jacekdmochowski/PROJECTS/fus/data/SUBJECTS/JD_FUS_meat/SESSIONS/1/ACQUISITIONS/rfMRI_300kPa/FILES'
filename = '1_3_12_2_1107_5_2_43_166037_2023121116481219261612190_0_0_0.nii.gz'

nii_3 = load_untouch_nii(fullfile(path_to_data,filename));
bold_3 = double(nii_3.img);

figure; 
subplot(221); imagesc(squeeze(mean(bold_3(:,:,10,:),4))); title('300 kPa (scan 1)','FontWeight','normal'); colorbar
subplot(222); imagesc(squeeze(mean(bold_3(:,:,30,:),4))); colorbar
subplot(223); imagesc(squeeze(mean(bold_3(:,:,50,:),4))); colorbar
subplot(224); imagesc(squeeze(mean(bold_3(:,:,70,:),4))); colorbar

jdprint('meat_phantom_bold_300kPa_scan1.png')

%% 300 kPa _1 
path_to_data = '/Users/jacekdmochowski/PROJECTS/fus/data/SUBJECTS/JD_FUS_meat/SESSIONS/1/ACQUISITIONS/rfMRI_300kPa_1/FILES'
filename = '1_3_12_2_1107_5_2_43_166037_2023121116503241597727038_0_0_0.nii.gz'

nii_3_1 = load_untouch_nii(fullfile(path_to_data,filename));
bold_3_1 = double(nii_3_1.img);

figure; 
subplot(221); imagesc(squeeze(mean(bold_3_1(:,:,10,:),4))); title('300 kPa (scan 2)','FontWeight','normal')
colorbar
subplot(222); imagesc(squeeze(mean(bold_3_1(:,:,30,:),4))); colorbar
subplot(223); imagesc(squeeze(mean(bold_3_1(:,:,50,:),4))); colorbar
subplot(224); imagesc(squeeze(mean(bold_3_1(:,:,70,:),4))); colorbar

jdprint('meat_phantom_bold_300kPa_scan2.png')




