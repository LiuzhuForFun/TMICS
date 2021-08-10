clear
close all
count = 1;
% nums=[168,116,125,298,256,250,219,250];
for i = 1:25
    for j = 3:29
%         input_name = sprintf('./NTU/%d-c-%d.jpg',i,j);
        input_name = sprintf('./%d/result-d-07-%d.jpg',i,j);
        %img_path = sprintf('./%s/*.jpg',num2str(i));
        %input_name = sprintf(dir(img_path)(j).name);
		input = im2double(imread(input_name));
        gt_name = sprintf('/media/liuzhu/somedata/frames_heavy_test_JPEG/%d/gtc-%d.jpg',i,j);
%         gt_name = sprintf('/media/liuzhu/somedata/Dataset_Testing_Synthetic/GTC/%d/%d.jpg',i,j);
%         gt_name = sprintf('./frames_light_test_JPEG/%d/gtc-%d.jpg',i,j);
		%gt_name = sprintf('./gtc/%d/gtc-%d.jpg',i,j);
        gt = im2double(imread(gt_name));
%         [w,h,~] = size(gt); ��ȷ��[h, w, c]
%         size(gt)
%         fprintf('img_name:%s\n',input_name)
%         fprintf('gt_name:%s\n',gt_name)
        
        ycbcr = rgb2ycbcr(input);
        input_gray = ycbcr(:,:,1);
        ycbcr = rgb2ycbcr(gt(:, :, :));
            y1 = gt(:, :, :);
        if i >= 10
            ycbcr = rgb2ycbcr(gt(1:500, 1:888, :));    % ��Ϊ��������е���w=888������889��gtc��w=889
            y1 = gt(1:500, 1:888, :);
        else
            ycbcr = rgb2ycbcr(gt(:, :, :));
            y1 = gt(:, :, :);
        end
        gt_gray = ycbcr(:,:,1);
        % 35.1618/0.9380
         PSNR(count) = psnr(input_gray,gt_gray);
%          SSIM(count) = ssim(input_gray,gt_gray);
%         PSNR(count) = psnr(input,y1);
         SSIM(count) = ssim(input_gray,gt_gray);
        count = count + 1;
        fprintf('i:%d, j:%d\n',i,j);
    end
end
PSNR_ave = mean(PSNR);
SSIM_ave = mean(SSIM);