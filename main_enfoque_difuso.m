close all
clear
clc

i = 1; % Indice de imagen a analizar de BSDS500, seleccionar en el rango [1, 500]


%% Carga de base de datos BSDS500
mkdir('Recursos')
addpath(genpath('Recursos'))
imds_train = imageDatastore('Recursos/BSDS500/data/images/train');
imds_val = imageDatastore('Recursos/BSDS500/data/images/val');
imds_test = imageDatastore('Recursos/BSDS500/data/images/test');
[GT_train, list_train] = read_data_BSDS500('train');
[GT_val, list_val] = read_data_BSDS500('val');
[GT_test, list_test] = read_data_BSDS500('test');

imds = [imds_train.Files; imds_val.Files; imds_test.Files];
GroundT = [GT_train; GT_val; GT_test];

num_pixeles = 154401;
I = func_Im_norm(imread(imds{i}),num_pixeles);

%% Segmentación por el algoritmo SFFCM (este proceso no forma parte de la evaluación)
SP = 400;
C = 5;
S = FFCM(I, SP, C);

%% Complejidad de imagen
% Carga de recursos para el uso de la ANN
cross1 = load('Cross1_154k_0.69_28.mat');
cross2 = load('Cross2_154k_0.95_28.mat');
cross3 = load('Cross3_154k_0.86_28.mat');
cross4 = load('Cross4_154k_0.88_28.mat');
cross5 = load('Cross5_154k_0.78_28.mat');
cross6 = load('Cross6_154k_0.87_28.mat');
cross7 = load('Cross7_154k_0.72_28.mat');

% Extracción de características
%******************************************************
tic
I_features = FeaturesCorchs(I);

Features(1) = I_features.Contrast; Features(2) = I_features.Correlation; Features(3) = I_features.Energy;
Features(4) = I_features.Homogeneity; Features(5) = I_features.FrequencyFactor; Features(6) = I_features.EdgeDensity;
Features(7) = I_features.CompressionRatio; Features(8) = I_features.NumberOfRegions; Features(9) = I_features.Colorfulness;
Features(10) = I_features.NumberOfColors; Features(11) = I_features.ColorHarmony;

for j = 1:11
    Features_N1(j) = (Features(j)-cross1.Fmin(j))/(cross1.Fmax(j)-cross1.Fmin(j)); 
    Features_N2(j) = (Features(j)-cross2.Fmin(j))/(cross2.Fmax(j)-cross2.Fmin(j)); 
    Features_N3(j) = (Features(j)-cross3.Fmin(j))/(cross3.Fmax(j)-cross3.Fmin(j)); 
    Features_N4(j) = (Features(j)-cross4.Fmin(j))/(cross4.Fmax(j)-cross4.Fmin(j)); 
    Features_N5(j) = (Features(j)-cross5.Fmin(j))/(cross5.Fmax(j)-cross5.Fmin(j)); 
    Features_N6(j) = (Features(j)-cross6.Fmin(j))/(cross6.Fmax(j)-cross6.Fmin(j)); 
    Features_N7(j) = (Features(j)-cross7.Fmin(j))/(cross7.Fmax(j)-cross7.Fmin(j)); 
end

% PCA
PCA_features1 = processpca('apply',Features_N1',cross1.PS); PCA_features1(6:11) = [];
PCA_features2 = processpca('apply',Features_N2',cross2.PS); PCA_features2(6:11) = [];
PCA_features3 = processpca('apply',Features_N3',cross3.PS); PCA_features3(6:11) = [];
PCA_features4 = processpca('apply',Features_N4',cross4.PS); PCA_features4(6:11) = [];
PCA_features5 = processpca('apply',Features_N5',cross5.PS); PCA_features5(6:11) = [];
PCA_features6 = processpca('apply',Features_N6',cross6.PS); PCA_features6(6:11) = [];
PCA_features7 = processpca('apply',Features_N7',cross7.PS); PCA_features7(6:11) = [];

for i = 1:7
    eval_net = strcat('O(',num2str(i),',:) = cross',num2str(i),'.net(PCA_features',num2str(i),');');
    eval(eval_net)
end

PSI = mean(O);
tiempo_complejidad = toc
%******************************************************

%% Evaluación por métricas M
%******************************************************
tic
GT = GroundT(i);   
PRI = [];
VI = [];
GCE = [];
BDE = [];
for benchIndex=1:length(GT.groundTruth)
    benchLabels = GT.groundTruth{1,benchIndex}.Segmentation;
    [curRI,curGCE,curVOI] = compare_segmentations(S(:,:,1),benchLabels);       
    [curBDE, ~] = compare_image_boundary_error(double(S(:,:,1)), double(benchLabels));

    PRI(benchIndex) = curRI;
    VI(benchIndex) = curVOI;
    GCE(benchIndex) = curGCE;
    BDE(benchIndex) = curBDE;

end
M_PRI = mean(PRI);
M_VI = mean(VI);
M_GCE = mean(GCE);
M_BDE = mean(BDE);
tiempo_M = toc
%******************************************************

%% Fusificación de métricas M
%******************************************************
tic
% user = memory;
u_PRI = muPRI(M_PRI, PSI);
u_VI = muVI(M_VI, PSI);
u_GCE = muGCE(M_GCE, PSI);
u_BDE = muPRI(M_BDE, PSI);
% disp(user.MemUsedMATLAB)
tiempo_fusificacion = toc
%******************************************************

%% Agregación
%******************************************************
tic 

user = memory;
u_SEG_H = harmmean([u_PRI u_VI u_GCE u_BDE]);
disp(user.MemUsedMATLAB)

tiempo_agregacion = toc
%******************************************************









%% Funciones

function [GT, listing] = read_data_BSDS500(particion)
    data_dir = ['Recursos/BSDS500/data/groundTruth/'  particion];
    cd(data_dir);
    listing = dir('*.mat');
    cd('../../../../../');
    GT = [];
    for i = 1:length(listing)
        aux = load(strcat(data_dir,'/',listing(i).name));
        GT = [GT; aux];
    end  
end