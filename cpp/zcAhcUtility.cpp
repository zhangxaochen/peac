#include "zcAhcUtility.h"

#include <fstream>

#include <pcl/pcl_macros.h> //zc: RAD2DEG

using std::cout;
using std::endl;
using std::ofstream;

//@brief 输出 t3+R9 到
void processPoses(const char *fn, const Affine3dVec &poses){
	ofstream fout(fn);
	for(size_t i=0; i<poses.size(); i++){
		const Affine3d &pose = poses[i];
		Eigen::Quaterniond q (pose.rotation ());
		Eigen::Vector3d t (pose.translation ());
		//呼应前面策略, 若 t 很大, 认为是无效值FLAG, csv中整行 //2016-12-8 15:05:41
		float invalidThresh = 9e2;
		if(t[0] > invalidThresh){
			fout << "0,0,0,0,0,0,0" <<endl;
		}
		else{
			fout << t[0] << "," << t[1] << "," << t[2]
			<< "," << q.w () << "," << q.x () << "," << q.y ()<< ","  << q.z () << std::endl;
		}
	}
	fout.close();
}//processPose

//@brief 输出平面参数: 法向,质心,曲率, mse(这啥用?)
void printPlaneParams(const double *normals, const double *center, double curvature, double mse){
	printf("normal=(%f,%f,%f); center=(%f,%f,%f); curv=%f; mse=%f\n", normals[0],normals[1],normals[2], center[0],center[1],center[2], curvature, mse);
}

void printPlaneParams(const PlaneSeg &planeSeg){
	printPlaneParams(planeSeg.normal, planeSeg.center, planeSeg.curvature, planeSeg.mse);
}

//@brief 点乘, 向量长度不固定
double dotProd(const double *v1, const double *v2, size_t len /*= 3*/){
	double sum = 0;
	for(size_t i=0; i<len; i++){
		sum += (v1[i]*v2[i]);
	}
	return sum;
}//dotProd

//@brief 向量模长
double norm(const double *v, size_t len/* = 3*/){
	return sqrt(dotProd(v, v, len));
}//norm

//@brief 两点欧氏距离
double dist(const double *v1, const double *v2, size_t len /*= 3*/){
	double sum = 0;
	for(size_t i=0; i<len; i++){
		double v12sub = v1[i] - v2[i];
		sum += v12sub * v12sub;
	}
	sum = sqrt(sum);
	return sum;
}//dist

//@brief 叉积 (v1 x v2)=v3; 只管三维3D;
//@param[out] v3, 输出叉积结果; 需要外部预分配内存
void crossProd(const double *v1, const double *v2, double *v3){
	//见: http://baike.baidu.com/view/973423.htm
	v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
	v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
	v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
}//crossProd

//@brief 施密特正交化, 目前只管前两轴, 第三轴这里不管(外部直接做叉积); 默认向量也是3D的; 输入不必是单位向量
//@param[in] v1, 主参考轴
//@param[out] newv1: =v1/|v1| 单位化; 需要外部预分配内存
//@param[out] newv2: v2 参照 v1 正交化之后的输出; 需要外部预分配内存
void schmidtOrtho(const double *v1, const double *v2, double *newv1, double *newv2, size_t len /*= 3*/){
	//见: http://blog.csdn.net/mathmetics/article/details/21444077
	//1. v1单位化
	double v1norm = norm(v1, len);
	for(size_t i=0; i<len; i++)
		newv1[i] = v1[i] / v1norm;

	//2. v2正交化, 
	double v2proj = dotProd(v2, newv1, len); //v2 在 v1方向的投影模长
	for(size_t i=0; i<len; i++)
		newv2[i] = v2[i] - v2proj * newv1[i];
	//2.2 v2 单位化
	double newv2norm = norm(newv2, len);
	for(size_t i=0; i<len; i++)
		newv2[i] = newv2[i] / newv2norm;
}//schmidtOrtho


//@brief 列主元高斯消去法; 见: https://www.oschina.net/code/snippet_76_4375
//A为系数矩阵，x为解向量，若成功，返回true，否则返回false，并将x清空。
bool RGauss(const vector<vector<double> > & A, vector<double> & x){
	x.clear();
	//由于输入函数已经保证了系数矩阵是对的，就不检查A了
	int n = A.size();
	int m = A[0].size();
	x.resize(n);
	//复制系数矩阵，防止修改原矩阵
	vector<vector<double> > Atemp(n);
	for (int i = 0; i < n; i++)
	{
		vector<double> temp(m);
		for (int j = 0; j < m; j++)
		{
			temp[j] = A[i][j];
		}
		Atemp[i] = temp;
		temp.clear();
	}
	for (int k = 0; k < n; k++)
	{
		//选主元
		double max = -1;
		int l = -1;
		for (int i = k; i < n; i++)
		{
			if (abs(Atemp[i][k]) > max)
			{
				max = abs(Atemp[i][k]);
				l = i;
			}
		}
		if (l != k)
		{
			//交换系数矩阵的l行和k行
			for (int i = 0; i < m; i++)
			{
				double temp = Atemp[l][i];
				Atemp[l][i] = Atemp[k][i];
				Atemp[k][i] = temp;
			}
		}
		//消元
		for (int i = k+1; i < n; i++)
		{
			double l = Atemp[i][k]/Atemp[k][k];
			for (int j = k; j < m; j++)
			{
				Atemp[i][j] = Atemp[i][j] - l*Atemp[k][j];
			}
		}
	}
	//回代
	x[n-1] = Atemp[n-1][m-1]/Atemp[n-1][m-2];
	for (int k = n-2; k >= 0; k--)
	{
		double s = 0.0;
		for (int j = k+1; j < n; j++)
		{
			s += Atemp[k][j]*x[j];
		}
		x[k] = (Atemp[k][m-1] - s)/Atemp[k][k];
	}
	return true;
}//RGauss

//@brief 其实是修改 pSegMat, 并不 imshow
//@param[in] labelMat: 如 ahc.PlaneFitter.membershipImg 
//@param[in] pSegMat: 调试观察mat; 为啥用指针: 其实用传值也行, 但是不能传引用, 否则: cannot bind non-const lvalue reference of type 'int&' to an rvalue of type 'int'
//void showLabelMat(cv::Mat labelMat, cv::Mat *pSegMat /*= 0*/){
void annotateLabelMat(cv::Mat labelMat, cv::Mat *pSegMat /*= 0*/){ //改名
	using namespace cv;
	int *lbMatData = (int*)labelMat.data; //label mat raw data
	int matEcnt = labelMat.rows * labelMat.cols; //mat elem count
	set<int> lbSet(lbMatData, lbMatData + matEcnt); //label set, 居然有 -6~-1, 什么含义? 不懂! //2016-9-13 22:39:42
	int labelCnt = lbSet.size();
	//if(dbg2_)
		printf("labelCnt= %d\n", labelCnt);
	for(set<int>::const_iterator iter=lbSet.begin(); iter!=lbSet.end(); iter++){
		int currLabel = *iter;
		if(currLabel < 0)
			continue;
		Mat msk = (labelMat == currLabel);
		// 		Mat erodeKrnl = getStructuringElement(MORPH_RECT, Size(5,5));
		// 		erode(msk, msk, erodeKrnl); //腐蚀msk, 以便消除噪点, 仅保留一个主要contour //不好使, 改用找最大轮廓

		vector<vector<Point> > contours;
		findContours(msk, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		int maxContNum = 0;
		size_t maxCntIdx = 0; //最大contour的idx
		for(size_t i=0; i<contours.size(); i++){ //找最大轮廓
			if(contours[i].size() > maxContNum){
				maxContNum = contours[i].size();
				maxCntIdx = i;
			}
		}

		Rect bdRect = boundingRect(contours[maxCntIdx]);

		Scalar colorRed(0,0,255);
		Scalar colorWhite(255,255,255);
		//rectangle(*pSegMat, bdRect, colorWhite, 2);
		//putText(*pSegMat, std::to_string(long long(currLabel)), bdRect.tl(), FONT_HERSHEY_SIMPLEX, 1, colorWhite, 2);

		//+++++++++++++++改用多种 color, 否则肉眼难看 //7种, 非8种, 没黑色 //2016-12-27 11:04:11
		//const Scalar colorArr[] = {Scalar(0,0,255), Scalar(0,255,0), Scalar(0,255,255),
		//	Scalar(255,0,0), Scalar(255,0,255), Scalar(255,255,0), Scalar(255,255,255)};
		const Scalar colorArr[] = {Scalar(128,128,255), Scalar(128,255,128), Scalar(128,255,255),
			Scalar(255,128,128), Scalar(255,128,255), Scalar(255,255,128), Scalar(255,255,255)};

		static int cidx = -1;
		cidx++;
		cidx = cidx % 7;
		//rectangle(*pSegMat, bdRect, colorArr[cidx], 1);
// 		putText(*pSegMat, std::to_string(long long(currLabel)), bdRect.tl(), FONT_HERSHEY_SIMPLEX, 1, colorArr[cidx], 2);
// 		putText(*pSegMat, std::to_string(long long(currLabel)), bdRect.br(), FONT_HERSHEY_SIMPLEX, 1, colorArr[cidx], 2);

		//Point bdRectCenter(bdRect.x + bdRect.width / 2, bdRect.y + bdRect.height / 2);
// 		Point bdRectAnchor(bdRect.x + bdRect.width * 1. / 3, bdRect.y + bdRect.height * 2. / 3); //不放中间, 改放在偏左下角 1/3 处
// 		putText(*pSegMat, std::to_string(long long(currLabel)), bdRectAnchor, FONT_HERSHEY_SIMPLEX, 1, colorArr[cidx], 2); //也绘制在矩形框中心
        //↑--上面也不好看, 还是改用质心--↓  //2017-2-21 14:53:00
        Moments mu = moments(contours[maxCntIdx]);
        Point mcen(mu.m10/mu.m00, mu.m01/mu.m00);
		//putText(*pSegMat, std::to_string(long long(currLabel)), mcen, FONT_HERSHEY_SIMPLEX, 1, colorArr[cidx], 2); //在轮廓质心
	}//for-lbSet
}//annotateLabelMat


//@brief zc: 通过 lblMat, 将未确定/无方向的三轴直线, 确定为三轴射线(有方向)
//@param[in] dmap, 原深度图, 量纲mm
//@param[in] orig, 原点, 即 t3; 量纲m
//@param[in] axs, 三轴直线, 即 R9; 量纲无, 三个单位向量; 可能输入时已经是右手坐标系, 但返回值输出可能破坏此属性
vector<double> zcAxLine2ray(const cv::Mat &dmap, const vector<double> &orig, const vector<double> &axs,
	double fx, double fy, double cx, double cy){
	vector<double> resAxs;

	const double STEP = 10; //沿轴向漫步的步长
	double ox = orig[0],
			oy = orig[1],
			oz = orig[2];
	for(size_t i=0; i<3; i++){
		double axx = axs[i * 3 + 0];
		double axy = axs[i * 3 + 1];
		double axz = axs[i * 3 + 2];
		
		double vx = ox + STEP*axx, //+STEP
			vy = oy + STEP*axy,
			vz = oz + STEP*axz;
		int u = (vx * fx) / vz + cx,
			v = (vy * fy) / vz + cy;
		ushort depth = dmap.at<ushort>(u, v);

		double vx_1 = ox - STEP*axx, //-STEP
			vy_1 = oy - STEP*axy,
			vz_1 = oz - STEP*axz;
		int u_1 = (vx_1 * fx) / vz_1 + cx,
			v_1 = (vy_1 * fy) / vz_1 + cy;
		ushort depth_1 = dmap.at<ushort>(u_1, v_1);

		int FLAG = abs(vz - depth) < abs(vz_1 - depth_1) ? 1 : -1;
		
		resAxs.push_back(FLAG * axx);
		resAxs.push_back(FLAG * axy);
		resAxs.push_back(FLAG * axz);
	}//for-i-3

	return resAxs;
}//zcAxLine2ray

//@brief zc: 找两两正交的三个平面, 组成一个三元组 (可能有多组, e.g., 立方体一个竖直棱的上下顶角, 或干扰物造成), 
//示意图见: http://codepad.org/he35YCTh
//@return ortho3tuples, 是pl-idx-tup3, 不是实际平面参数; 后期可能用不到了
//@param[in] plvec
//@param[in] lblMat 平面划分 label 图, 如 ahc.PlaneFitter.membershipImg 
//@param[out] cubeCandiPoses, vec-vec, 初始必须是空, N*12(t3+R9), 【del】R9不是真的旋转矩阵, 甚至不正交【del】. 改成 R9是旋转矩阵(nearest orhto 解得), 按行存储(row-major)??? 【不确定】
//@param[out] prev 【放弃】
//@param[in/out] dbgMat, e.g.: pf.run 输出的调试观察 Mat
vector<vector<int>> zcFindOrtho3tup(const vector<PlaneSeg> &plvec, const cv::Mat &lblMat,
	double fx, double fy, double cx, double cy,
	vector<vector<double>> &cubeCandiPoses, cv::OutputArray dbgMat /*= cv::noArray()*/){

	using namespace cv;
	
	//1. 检查所有平面, 清点正交、平行关系, 既不正交也不平行的忽略
	size_t plcnt = plvec.size();
	double orthoThresh = 
		//0.0174524; //cos(89°), 小于此为正交 //因为依赖 pcl160, 而OpenNIGrabber用默认内参,导致点云不准, 正交面达不到此阈值
		0.0871557; //cos(85°) 放宽, //【用了雷昊内参之后, 发现非常好, 不过仍然放宽
	double paralThresh = 
		//0.999847695; //cos(1°), 大于此为平行
		0.996194698; //cos(5°) 放宽
	vector<set<int>> orthoMap(plcnt);  //正交关系表, 理论上只填充上三角, 即最后一个vec应是空; 【其实 vector<set<int>> 更合理, 暂不改
	vector<vector<int>> ortho3tuples; //找到的三元组放在这, 内vec 故意用 vec 不用 set, 更易索引; 内vec必然size=3
	vector<set<int>> paralMap(plcnt); //平行关系表
	for(size_t i=0; i<plcnt; i++){
		const PlaneSeg &pl_i = plvec[i];
		const double *norm_i = pl_i.normal;
		for(size_t j=i+1; j<plcnt; j++){ //从 i+1, 只看上三角
			const PlaneSeg &pl_j = plvec[j];
			const double *norm_j = pl_j.normal;
			double cosNorm = dotProd(norm_i, norm_j, 3); //因 |a|=|b|=1, 故直接 cos(a,b)=a.dot(b)

			//if(dbg2_)
				//printf("i,j=(%u,%u), cosNorm=%f; angle=%f\n", i, j, cosNorm, RAD2DEG(acos(cosNorm)));

			if(abs(cosNorm) < orthoThresh)
				orthoMap[i].insert(j);
			if(abs(cosNorm) > paralThresh)
				paralMap[i].insert(j);
		}//for-j
	}//for-i-plcnt

	//2. 在每个 orthoMap[i] 中, 若还有正交的, 则找到一个三元组! //理论上【已知立方体】上只能找到两个三元组, 若>2, 则要通过平行面间距做排除
	for(size_t i=0; i<orthoMap.size(); i++){
		set<int> &ortho_i = orthoMap[i]; //与id=i正交的平面id们
		if(ortho_i.size() < 2) //若 <2, 则构不成三元组
			continue;
		set<int>::const_iterator iter_j = ortho_i.begin();
		for(; iter_j!=ortho_i.end(); iter_j++){
			int idx_j = *iter_j;
			set<int> &ortho_j = orthoMap[idx_j]; //即将在此找 idx_k, 若找到, 则成一个三元组
			if(ortho_j.size() == 0)
				continue;
			//set<int>::const_iterator iter_k = iter_j + 1; //×, 无 '+'
			//set<int>::const_iterator iter_k = iter_j; iter_k++; //√, 不过原始
			set<int>::const_iterator iter_k = std::next(iter_j);
			for(; iter_k!=ortho_i.end(); iter_k++){
				int idx_k = *iter_k;
				if(ortho_j.count(idx_k)){ //找到三元组
					vector<int> tuple3;
					tuple3.push_back(i);
					tuple3.push_back(idx_j);
					tuple3.push_back(idx_k);
					ortho3tuples.push_back(tuple3);
				}
			}//for-iter_k
		}//for-iter_j
	}//for-i-orthoMap

	//3. 在 ortho3tuples 可能存在【假的】三元组, 判定依据: 三面实际相邻才算三元组-->进化为: 【放弃】三面交点(3D)在labelMat(2D)上某len邻域内有三种有效label【放弃】
	//判定依据改为: 邻域label-set 包含 tuple3-vec (虽然一个 set 一个 vector, 用 std::includes 算法) //2016-12-7 20:55:25
	vector<vector<int>> tmp3tuples;
	for(size_t i=0; i<ortho3tuples.size(); i++){
		vector<int> &tuple3 = ortho3tuples[i];
		//1. 取三面, 构造 Ax=b 的 [A|b] 增广矩阵; 列主元高斯, 求解三面顶点
		vector<vector<double>> matA;
		vector<double> matb;
		for (int ii=0; ii<3; ii++){
			int plIdx = tuple3[ii];
			const PlaneSeg &plseg = plvec[plIdx];
			//平面参数ABCD: (ABC)=normal; D= -dot(normal, center) //注意负号, b[i]=-D=dot...
			vector<double> tmpRow(plseg.normal, plseg.normal+3);
			double b_i = dotProd(plseg.normal, plseg.center);
			tmpRow.push_back(b_i); //系数矩阵一行, 包含 Ai|bi
			matA.push_back(tmpRow);
		}
		vector<double> vertx; //三面交点, 方程组的解; 尺度是不是米(m)啊？ √
		RGauss(matA, vertx);

		//2. 顶点的像素邻域内是否有三种有效label
		//3D->2D像素点, 是int, 不追求精度
		int ou = (vertx[0] * fx) / vertx[2] + cx,
			ov = (vertx[1] * fy) / vertx[2] + cy;

		//int winSz = 20 / qvgaFactor_; //邻域窗口长度
		int winSz = 20; //抽成函数后, 暂时不要 qvgaFactor_ 了

		cv::Rect tmpRoi(ou - winSz/2, ov - winSz/2, winSz, winSz);
		if(tmpRoi != (tmpRoi & cv::Rect(0,0, lblMat.cols, lblMat.rows)) ) //若邻域不在图像范围内, 忽略
			continue;
		//else: 否则, 若整个邻域小方框都在图像内, 继续
		//if(dbg1_){
		if(! dbgMat.empty()){ //改成不用全局 flag
			cv::circle(dbgMat.getMatRef(), cv::Point(ou, ov), 2, 255, 1); //蓝小圆圈
			cv::circle(dbgMat.getMatRef(), cv::Point(ou, ov), 7, cv::Scalar(0,0,255), 2); //红大圆圈, 同心圆, 调试观察便利
		}

		//cv::Mat vertxNbrLmat(pf.membershipImg, tmpRoi); //邻域 label-mat
		cv::Mat vertxNbrLmat(lblMat, tmpRoi); //邻域 label-mat
		
		vertxNbrLmat = vertxNbrLmat.clone(); //clone 能解决set(labelMat) 出错问题吗? √ 能!
				//记不清了, 应该是说, 若不 clone, 则roi仅是view而非连续内存, 取 [data, data+size] 时会取到原 Mat 一行片段(如 1*16), 而非真正需要的方块(如4*4) //2016-12-28 11:22:14
		//cout<<"vertxNbrLmat:\n"<<vertxNbrLmat<<endl;
		//cv::Mat dbgRoiMat(seg, tmpRoi); //调试观察小区域

		int *vertxNbrLmatData = (int*)vertxNbrLmat.data; //label mat raw data
		set<int> nbrLabelSet(vertxNbrLmatData, vertxNbrLmatData + winSz * winSz);
		//int posCnt = 0; //邻域 label>0 统计量
		//for(set<int>::const_iterator it = nbrLabelSet.begin(); it != nbrLabelSet.end(); it++){
		//	if(*it >= 0) //0也是有效label
		//		posCnt++;
		//}
		if(std::includes(nbrLabelSet.begin(), nbrLabelSet.end(), tuple3.begin(), tuple3.end()) ){ //已验证, 效果很好 2016-12-9 00:09:40
		//if(posCnt >= 3){ //认定为真实正交三邻面三元组
			//if(dbg1_){
			if(! dbgMat.empty()){ //改成不用全局 flag
				cv::circle(dbgMat.getMatRef(), cv::Point(ou, ov), 2, 255, 1); //蓝小圆圈
				cv::circle(dbgMat.getMatRef(), cv::Point(ou, ov), 7, cv::Scalar(0,255,0), 2); //绿大圆圈, 同心圆, 调试观察便利, //表示筛选最终定下的顶角
			}

			tmp3tuples.push_back(tuple3);
			cubeCandiPoses.push_back(vertx); //先把(R,t)的t填充; 之后下面 cubePoses 不要 push, 要在每行 .insert(.end, dat, dat+3);

			//改动: cubeCandiPoses 在此 if块内, 一并填充, 应该合理 //2016-12-29 11:14:27
			vector<double> ax3orig; //初始三轴, 不完美正交
			ax3orig.reserve(9);
			//v2: 最优正交化, 见: https://www.evernote.com/shard/s399/nl/67976577/48135b5e-7209-47c1-9330-934ac4fee823
#if 01	//v2.1 三面【法向】做轴, 不正交没关系
			for(size_t kk=0; kk<3; kk++){
				const double *pl_k_norm = plvec[tuple3[kk]].normal;
				ax3orig.insert(ax3orig.end(), pl_k_norm, pl_k_norm+3);
			}
#endif //法向/交线 谁做初始轴

			//最近正交矩阵问题: 
			//https://en.wikipedia.org/wiki/Orthogonal_matrix#Nearest_orthogonal_matrix
			//http://math.stackexchange.com/questions/571817/given-three-vectors-how-to-find-an-orthonormal-basis-closest-to-them
			JacobiSVD<Matrix3d> svd(Map<Matrix3d>(ax3orig.data()), ComputeFullU | ComputeFullV);
			Matrix3d svdU = svd.matrixU();
			Matrix3d svdV = svd.matrixV();
			Matrix3d orthoAxs = svdU * svdV.transpose(); //这里得到关于 ax3orig 的最优化正交基, det=±1, 不确保是旋转矩阵

			//这里跟之前非函数代码段不同: //2016-12-29 20:57:59
			//之前用策略 PROCRUSTES 时, cubeCandiPoses 中存的是 ax3orig, @L930~940; 现在想改一改
			//正交阵变旋转矩阵:
#if 01
			int FLAG = orthoAxs.determinant() > 0 ? 1 : -1; //det 理论上是 ±1, 可直接作为 flag, 但保险起见...
			//orthoAxs.row(2) *= FLAG; //错 ×
			orthoAxs.col(2) *= FLAG; //陷阱: ↑-上面是 row内存按col放入的, 所以这里应该 .col(2); //太 trick, 暂时这样
#else
			if(orthoAxs.determinant() < 0)
				//todo-交换某两列-suspend...
				;
#endif

			double *axs = orthoAxs.data();
			//axs = orthoAxs.transpose().data(); //为啥要 .T? 因为 ax3orig 按行存, 但按列放入svd, 这里又是按列取出; 若想保持与 ax3orig 近似, 要转置
				//此处转置虽然影响轴, 不管用, 关键在前面 .col(
				//且理论上也错了！不应 .T！ 因为: 本来按列入, 按列取出, 仍是 ax3orig 按行存储 //2016-12-30 01:05:06

			//调试绘制三轴
			if(! dbgMat.empty() && 0 == i){
				const Scalar colorCmy[] = {Scalar(255,255,0),
											Scalar(255,0,255),
											Scalar(0,255,255)};
				for(size_t i=0; i<3; i++){
					//应该错了, 但画出来好像对了
// 					double x = axs[i*3+0],
// 							y = axs[i*3+1],
// 							z = axs[i*3+2];
					double axLen = 0.30; //绘制轴的长度, 量纲m
					double x = axLen * axs[i*3+0] + vertx[0],
							y = axLen * axs[i*3+1] + vertx[1],
							z = axLen * axs[i*3+2] + vertx[2];

					//axes in 2D (u,v):
					int au = (int)(x / z * fx + cx),
						av = (int)(y / z * fy + cy);

					cv::line(dbgMat.getMatRef(), cv::Point(ou, ov), cv::Point(au, av), colorCmy[i]);
				}
			}

			vector<double> &currCrnr = cubeCandiPoses.back(); //tmp thing...
			currCrnr.insert(currCrnr.end(), axs, axs+9);
		}//if-nbrLabelSet-includes-tuple3
	}//for-ortho3tuples
	ortho3tuples.swap(tmp3tuples);

	return ortho3tuples;
}//zcFindOrtho3tup

Affine3d getCuPoseFromCandiPoses(const vector<vector<double>> &cubeCandiPoses){
	Affine3d res;

	// 	size_t crnrCnt = cubeCandiPoses.size();
	// 	for(size_t i=0; i<crnrCnt; i++){
	// 		vector<double> &rt = cubeCandiPoses[i];
	// 		rt[]
	// 	}

#if 1	//最懒策略, 拿 [0] 直接做立方体姿态
	vector<double> rt = cubeCandiPoses[0];
	res.translation() = Map<Vector3d>(rt.data());
	res.linear() = Map<Matrix3d>(rt.data()+3);
#endif

	return res;
}//getCuPoseFromCandiPoses

//vector<double> getCu4Pts(const vector<double> &crnrTR, const vector<float> &cuSideVec,const cv::Mat &dmap, cv::Mat &labelMat, float fx, float fy, float cx, float cy){
bool getCu4Pts(const vector<double> &crnrTR, const vector<float> &cuSideVec,const cv::Mat &dmap, cv::Mat &labelMat, float fx, float fy, float cx, float cy, vector<double> &pts4){

	//vector<double> res; //返回值: 1*12 vec, 4个坐标点 //改用 pts4
	Vector3d pt0(crnrTR.data()); //三邻面交点
	pts4.insert(pts4.end(), pt0.data(), pt0.data()+3);

	const float STEP = 0.01; //1cm, 10mm, 
	for(size_t i=1; i<=3; i++){ //对每一条轴
		Vector3d axi(crnrTR.data()+i*3); //三条棱边(可能实际由法向伪装)

		const int stepCnt = 5;
		Vector3d negaDirect = pt0 - axi * STEP * stepCnt, //稍稍增大 step
			posiDirect = pt0 + axi * STEP * stepCnt; //注意 +、-

		cv::Point px_n = getPxFrom3d(negaDirect, fx, fy, cx, cy);
		cv::Point px_p = getPxFrom3d(posiDirect, fx, fy, cx, cy);
#if 0   //dist_p & dist_n 距离判定不好, 当判定底面顶角时, 理论上二者值都应该很小
		float dist_n = abs(dmap.at<ushort>(px_n.y, px_n.x) / M2MM - negaDirect[2]);
		float dist_p = abs(dmap.at<ushort>(px_p.y, px_p.x) / M2MM - posiDirect[2]);

		int direct = dist_p < dist_n ? +1 : -1; //确定射线方向
		//float dist_ref = dist_p < dist_n ? dist_p : dist_n; //取小的作参考
#elif 1
		set<int> nbrLblSet_n, nbrLblSet_p;
		const int nbrCnt = 4;
		const int margin = 5;
		cv::Point nbrDelta[nbrCnt] = {cv::Point(-margin, 0), cv::Point(margin, 0), 
			cv::Point(0, -margin), cv::Point(0, margin)};
		cv::Rect matRegion(0, 0, labelMat.cols, labelMat.rows); //用于检测区域有效性

		for(size_t inbr=0; inbr<nbrCnt; inbr++){
			cv::Point nbrPxi_n = px_n + nbrDelta[inbr];
			cv::Point nbrPxi_p = px_p + nbrDelta[inbr];
			if(matRegion.contains(nbrPxi_n)) //若此邻点在图像区域内
				nbrLblSet_n.insert(labelMat.at<int>(nbrPxi_n) );
			else
				return false;

			if(matRegion.contains(nbrPxi_p))
				nbrLblSet_p.insert(labelMat.at<int>(nbrPxi_p) );
			else
				return false; //若邻域超出图像边界, 认为该边不完整, 即当前帧未找到4顶点
		}
		//还要抛掉无效 lbl (未必 -1, 只要 <0)
		//nbrLblSet_n.erase(-1);
		//nbrLblSet_p.erase(-1);
		set<int>::iterator iter;
		for(iter=nbrLblSet_n.begin(); iter!=nbrLblSet_n.end();){
			if(*iter < 0)
				nbrLblSet_n.erase(iter++);
			else
				++iter;
		}
		for(iter=nbrLblSet_p.begin(); iter!=nbrLblSet_p.end();){
			if(*iter < 0)
				nbrLblSet_p.erase(iter++);
			else
				++iter;
		}

		int direct = 0;
		set<int> nbrLblSet_k;
		if(nbrLblSet_n.size()>1 && nbrLblSet_p.size()<=1){
			direct = -1;
			//nbrLblSet_k = nbrLblSet_n;
		}
		else if(nbrLblSet_n.size()<=1 && nbrLblSet_p.size()>1){
			direct = 1;
			//nbrLblSet_k = nbrLblSet_p;
		}
		else{
			printf("++FUCK: nbrLblSet_n.size, nbrLblSet_p.size: %d, %d\n", nbrLblSet_n.size(), nbrLblSet_p.size());
			return false; //这种异常情况, 难以纠正, 一并算作未找到
		}

		//nbrLblSet_k改存射线起点邻域 label 并集：
		//报错: error C3892: '_Dest' : you cannot assign to a variable that is const
		//std::set_union(nbrLblSet_p.begin(), nbrLblSet_p.end(), nbrLblSet_n.begin(), nbrLblSet_n.end(), nbrLblSet_k.begin());
		nbrLblSet_k = nbrLblSet_p;
		nbrLblSet_k.insert(nbrLblSet_n.begin(), nbrLblSet_n.end());
#endif

		int k = stepCnt;
		float sideLen = 0;
		while(1){ //小碎步一直走
			k++;
			sideLen = k * STEP;
			Vector3d pt_k = pt0 + direct * sideLen * axi;
			cv::Point px_k = getPxFrom3d(pt_k, fx, fy, cx, cy);
#if 0   //距离判定不好
			float dist_k = abs(dmap.at<ushort>(px_k.y, px_k.x) / M2MM - pt_k[2]);
			printf("%f, ", dist_k);

			//if(dist_k > STEP) //直到射线不贴合深度图, 终止小碎步
			if(dist_k > STEP*2)
				break;
#elif 0 //nbrLblSet_k.size() < 2 判定也不好, 容易走过了太多, 导致例如 250&300分不清
			set<int> nbrLblSet_k;
			for(size_t inbr=0; inbr<nbrCnt; inbr++){
				cv::Point nbrPxi_k = px_k + nbrDelta[inbr];
				if(matRegion.contains(nbrPxi_k))
					nbrLblSet_k.insert(labelMat.at<int>(nbrPxi_k) );
			}
			//nbrLblSet_k.erase(-1);
			for(iter=nbrLblSet_k.begin(); iter!=nbrLblSet_k.end();){
				if(*iter < 0)
					nbrLblSet_k.erase(iter++);
				else
					++iter;
			}

			if(nbrLblSet_k.size() < 2)
				break;
#elif 1 //nbrLblSet_k 在 while 之前, direct 判定时就连带赋值
			bool walkEnd = false;
			set<int> nbrLblSet_tmp;
			for(size_t inbr=0; inbr<nbrCnt; inbr++){
				cv::Point nbrPxi_k = px_k + nbrDelta[inbr];
				if(matRegion.contains(nbrPxi_k)){
					int nbrLbl = labelMat.at<int>(nbrPxi_k);
					nbrLblSet_tmp.insert(nbrLbl);

					if(nbrLbl >=0 && nbrLblSet_k.count(nbrLbl) == 0){
						//↑-理论上四周应有X种label, 若发现有效的、新label, 则判定走到头了
						walkEnd = true;
						break;
					}
				}
				else
					return false;
			}
			for(iter=nbrLblSet_tmp.begin(); iter!=nbrLblSet_tmp.end();){
				if(*iter < 0)
					nbrLblSet_tmp.erase(iter++);
				else
					++iter;
			}
			if(nbrLblSet_tmp.size() <= 1)
				walkEnd = true;

			if(walkEnd)
				break;
#endif
		}
		//sideLen -= STEP;
		//printf("\n");

		//看看与哪条边长度相近:
		float minLenDiff = 10.f; //m, 默认初值 10m, 权当极大值
		size_t j_idx = 0;
		for(size_t j=0; j<3; j++){
			float sdLenDiff = abs(sideLen - cuSideVec[j]);
			if(sdLenDiff < minLenDiff){
				minLenDiff = sdLenDiff;
				j_idx = j;
			}
		}
		Vector3d pti = pt0 + direct * cuSideVec[j_idx] * axi;
		pts4.insert(pts4.end(), pti.data(), pti.data()+3);
	}//for-i-3三轴

	//return res;
	return true; //能走到这里必然 true
}//getCu4Pts

cv::Point getPxFrom3d(const Vector3d &pt3d, float fx, float fy, float cx, float cy){
	//vector<int> res;
	cv::Point res;

	double x = pt3d[0],
		y = pt3d[1],
		z = pt3d[2];

	//res.push_back((int)(x / z * fx + cx));
	//res.push_back((int)(y / z * fy + cy));
	res.x = (int)(x / z * fx + cx /*+ 0.5f*/); //用 round 而非 flooring //还是改回 flooring, 因为 px->pt3d 时并没有考虑 round
	res.y = (int)(y / z * fy + cy /*+ 0.5f*/);

	return res;
}//getPxFrom3d

cv::Point2f getPx2fFrom3d(const Vector3d &pt3d, float fx, float fy, float cx, float cy){
	cv::Point2f res;

	double x = pt3d[0],
		y = pt3d[1],
		z = pt3d[2];

	res.x = x / z * fx + cx;
	res.y = y / z * fy + cy;

	return res;
}//getPx2fFrom3d

cv::Mat zcRenderCubeDmap(const Cube &cube, float fx, float fy, float cx, float cy){
	using namespace cv;

	Mat res = Mat::zeros(WIN_HH, WIN_WW, CV_16UC1); //初始全黑

#if 1   //V1: 试图提高效率, 先生成 cu mask, 再仅在 mask 内部计算 //2017-1-15 00:28:47
	const vector<Vector3d> &cuVerts8_cam = cube.cuVerts8_;
	size_t vertCnt = cuVerts8_cam.size(); //==8, 但此处仍获取一下
	CV_Assert(vertCnt == 8);

	vector<Point> pxs8;
	for(size_t i=0; i<vertCnt; i++){
		Point pxi = getPxFrom3d(cuVerts8_cam[i], fx, fy, cx, cy);
		pxs8.push_back(pxi);
	}

	vector<Point> hull;
	convexHull(pxs8, hull); //默认 clockwise=false, returnPoints=true
	
	Mat cuMask = Mat::zeros(res.size(), CV_8UC1);
	fillConvexPoly(cuMask, hull, 255);
	//imshow("zcRenderCubeDmap-cuMask", cuMask); //形如: http://imgur.com/bXTx1VG

	const double INVALID_DEPTH = 1e11; //用极大值做无效值
	for(int v=0; v<res.rows; v++){
		for(int u=0; u<res.cols; u++){
			if(0 == cuMask.at<uchar>(v, u)) //无效区域跳过, 试图提高效率
				continue;

			double depth = INVALID_DEPTH;

			//由像素获得视线 view ray //视线射线一端是原点 000
			double x = (u - cx) / fx,
				   y = (v - cy) / fy,
				   z = 1;
			Vector3d vray(x, y, z);
			
			//6个面中, 每对相对面【最多】显示一个:
#if 0	//逻辑不很对, 仅当立方体基本在视野中央时才对; isVrayFacetIntersect 若用连线夹角和方法: 100~140ms; 改用水平射线法:4~11ms
			for(int fi=0; fi<3; fi++){
				//目前填充 facet 是按对面填充: 01, 23, 45; 此假设非通用, 暂定
				int which = cube.facetVec_[2*fi].center_.z() < cube.facetVec_[2*fi+1].center_.z() ? 2*fi : 2*fi+1;
#elif 1	//6面全检查, 180~220ms
			for(int fi=0; fi<6; fi++){
				int which = fi;
#endif
				Vector3d ptInters(-1,-1,-1); //暂定 <0 为无效值, 因为理论上不应位于相机后面
				bool isInters = cube.isVrayFacetIntersect(vray, which, ptInters);
				//double z = ptInters[2]; //交点z值
				//if(isInters && 0 <= z && z < depth)
				//    depth = z;
				if(isInters){
					double z = ptInters[2]; //交点z值
					if(0 <= z && z < depth)
						depth = z;
				}
			}//for-fi

			if(depth != INVALID_DEPTH)
				res.at<ushort>(v, u) = (ushort)(M2MM * depth + 0.5f); //round, 非 flooring
		}//for-u
	}//for-v
#endif

	return res;
}//zcRenderCubeDmap

bool isLinePlaneIntersect(const Vector3d &L0, const Vector3d &L, const Vector3d &plnorm, const Vector3d &plp0, Vector3d &ptInters){
	bool res = false;

	//d=(p0-l0)*n/(l*n)
	double fenmu = L.dot(plnorm); //分母
	if(abs(fenmu) < 1e-8) //分母为零, 线面平行或线在面内, 都算 false
		res = false;
	else{
		res = true;

		double fenzi = (plp0 - L0).dot(plnorm);
		double d = fenzi / fenmu;
		ptInters = d * L + L0; //解得交点
	}

	return res;
}//isLinePlaneIntersect

void zcDashLine(CV_IN_OUT cv::Mat& img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color){
    //参考资料:
    //http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    //http://stackoverflow.com/questions/20605678/accessing-the-values-of-a-line-in-opencv
    //http://answers.opencv.org/question/10902/how-to-plot-dotted-line-using-opencv/
    using namespace cv;

    LineIterator it(img, pt1, pt2, 8);            // get a line iterator
    for(int i = 0; i < it.count; i++,it++)
        //if ( i % 3 != 0 ){         // every 5'th pixel gets dropped, blue stipple line
        if(0 < i%10 && i%10 < 6){
            //(*it)[0] = 200;
            if(img.channels() == 1)
                img.at<uchar>(it.pos()) = color[0];
            else if(img.channels() == 3){
                Vec3b &pxVal = img.at<Vec3b>(it.pos());
                pxVal[0] = color[0];
                pxVal[1] = color[1];
                pxVal[2] = color[2];
            }
        }

}//zcDashLine

Cube::Cube(const Cube &cuOther, const Affine3d &affine){
	//8顶点:
	for(size_t i=0; i<8; i++){
		Vector3d verti = affine * cuOther.cuVerts8_[i];
		cuVerts8_.push_back(verti);
	}
	//6个面: //各面顶点分组: 0142/3675, 0253/1476, 0361/2574;
	for(size_t i=0; i<6; i++){
		Facet faceti = cuOther.facetVec_[i];
		addFacet(faceti.vertIds_);
	}
}//Cube-ctor

bool Cube::isLineFacetIntersect(const Vector3d &L0, const Vector3d &L, const Facet &facet, Vector3d &ptInters) const{
	bool res = isLinePlaneIntersect(L0, L, facet.normal_, facet.center_, ptInters);
	if(res){ //若跟平面相交, 再看交点是否在面片内
		const vector<int> &vertIds = facet.vertIds_;

#if 0	//此方案仅能检测点在凸多边形内; 若点在形内, 则连线各顶点, 夹角和= 360°; 若在形外则 <360°
		vector<Vector3d> pt2vertVec;
		vector<double> pt2vertNormVec;
		for(int i=0; i<4; i++){
			Vector3d pt2vi = cuVerts8_[vertIds[i]] - ptInters;
			pt2vertVec.push_back(pt2vi);
			pt2vertNormVec.push_back(pt2vi.norm());
		}

		double angRad = 0;
		const int iinc[4] = {1,2,3,0};
		for(int i=0; i<4; i++){
			angRad += acos(pt2vertVec[i].dot(pt2vertVec[iinc[i]]) / (pt2vertNormVec[i] * pt2vertNormVec[iinc[i]]) );
		}

		if(2*PI - angRad > 1e-8)
			res = false;
#elif 1	//见: http://blog.chinaunix.net/uid-30332431-id-5140349.html
		//又见笔记: 《判断点在多边形内》
		int c = 0; //奇偶性 flag
		const int iinc[4] = {1,2,3,0};
		for(int i = 0; i < 4; i++){
			Vector3d vi = cuVerts8_[vertIds[i]],
					 vj = cuVerts8_[vertIds[iinc[i]]];
			//暂定只看 XY平面:
			double ptx = ptInters[0],
					pty = ptInters[1];
			double vix = vi[0],
					viy = vi[1],
					vjx = vj[0],
					vjy = vj[1];

			if((viy > pty) != (vjy > pty)
				&& (ptx < (vjx - vix) * (pty - viy) / (vjy - viy) + vix))
				c = !c; //1-odd-面片内; 0-even-面片外
		}
		res = c; //int->bool
#endif
	}//if(res)//若跟平面相交, 再看交点是否在面片内

	return res;
}//Cube::isLineFacetIntersect

void Cube::addFacet(const vector<int> &vertIds){
	const int fvertCnt = 4;
	CV_Assert(vertIds.size() == fvertCnt); //矩形面片, 必有四顶点

	//1, 构造 facet
	Facet tmpFacet;
	tmpFacet.vertIds_ = vertIds;

	//内部重新计算法向量:
	Vector3d edge1 = cuVerts8_[vertIds[1]] - cuVerts8_[vertIds[0]],
			 edge2 = cuVerts8_[vertIds[3]] - cuVerts8_[vertIds[0]];
	Vector3d plnorm = edge1.cross(edge2); //叉积叉乘
	plnorm.normalize();
	tmpFacet.normal_ = plnorm;

	//质心: 对角线中点
	tmpFacet.center_ = (cuVerts8_[vertIds[0]] + cuVerts8_[vertIds[2]]) / 2;

	facetVec_.push_back(tmpFacet);

#if 0
	//2, 填充 vertAdjFacetIds_
	if(vertAdjFacetIds_.size() == 0)
		vertAdjFacetIds_.resize(8);
	size_t facetId = facetVec_.size() - 1; //此面片id=0~7
	for(size_t i=0; i<fvertCnt; i++)
		vertAdjFacetIds_[i].insert(facetId);
#endif
}//Cube::addFacet

void Cube::addEdgeId(vector<int> &edge){
	edgeIds_.push_back(edge);
}//Cube::addEdgeId

void Cube::drawContour(cv::Mat dstCanvas, double fx, double fy, double cx, double cy, const cv::Scalar& color, bool hideLines /*= false*/){
    using namespace cv;
	if(dstCanvas.empty())
		dstCanvas = Mat::zeros(WIN_HH, WIN_WW, CV_8UC3); //若无内容, 生成待绘制黑背景, 尺寸默认 (WIN_HH, WIN_WW)
	
	//1, 找出要被消隐的顶点
	const size_t vertCnt = this->cuVerts8_.size(); //应==8, 这里仍用 .size() 获取, 以防有错
	CV_Assert(vertCnt == 8);

    set<int> occVertIds; //被遮挡顶点集合, 其邻边虚线绘制或不显示
    //对于每一个顶点
	for(size_t i=0; i<vertCnt; i++){
		Vector3d vi = cuVerts8_[i];
        //判断此顶点是否被最前方的三邻面所遮挡, 
        //即此三面, 且若不包含此顶点, 若视线与面片相交, 则遮挡

        //6个面中, 每对相对面【最多】显示一个:
        for(int fi=0; fi<3; fi++){
            //目前填充 facet 是按对面填充: 01, 23, 45; 此假设非通用, 暂定
            int which = facetVec_[2*fi].center_.z() < facetVec_[2*fi+1].center_.z() ? 2*fi : 2*fi+1;
            //略过包含此顶点的面片, (此情形是可能的)
            if(facetVec_[which].isContainVert(i))
                continue;

            bool isInters = isVrayFacetIntersect(vi, which);
            if(isInters){
                occVertIds.insert(i);
                break;
            }
        }//for-fi-3
	}//for-vertCnt

    //hidLineSet 消隐线段合集, 内set均包含两个端点, 用来表示线段
    //绘制策略: 若已画虚线, 则仅增画实线;
    set<set<int>> hidLineSet;
	const size_t facetCnt = this->facetVec_.size(); //应==6, 也仍用 .size() 获取, 以防有错
	for(size_t i=0; i<facetCnt; i++){
		Facet &fi = facetVec_[i]; //面片 i
        vector<int> &vidvec = fi.vertIds_; //面上顶点
        const size_t vcnt = vidvec.size(); //应==4, 也仍用 .size() 获取, 以防有错

        //判断是否面片被遮挡, 其只要存在一个顶点被遮挡, 则整面也被遮挡
        bool isFaceOccluded = false;
        for(size_t fvId=0; fvId<vcnt; fvId++){
            if(occVertIds.count(vidvec[fvId]) > 0){
                isFaceOccluded = true;
                break;
            }
        }

		for(size_t j=0; j<vcnt; j++){
            size_t j1 = (j+1) % vcnt;
			Vector3d vertj = cuVerts8_[vidvec[j]],
                     vertj1 = cuVerts8_[vidvec[j1]];

            //判断线段是否需要重绘, 若已
            set<int> lineSeg;
            lineSeg.insert(vidvec[j]);
            lineSeg.insert(vidvec[j1]);
            if(isFaceOccluded && hidLineSet.count(lineSeg) > 0)
                continue;

            Point ptj(vertj.x() / vertj.z() * fx + cx,
                      vertj.y() / vertj.z() * fy + cy);
            Point ptj1(vertj1.x() / vertj1.z() * fx + cx,
                      vertj1.y() / vertj1.z() * fy + cy);

            if(!isFaceOccluded)
                cv::line(dstCanvas, ptj, ptj1, color); //若8uc3, 则蓝色; 若 8uc1 则白色
            else if(!hideLines){ //若不消隐, 则绘制虚线
                zcDashLine(dstCanvas, ptj, ptj1, color);
                //且登记此条虚线
                hidLineSet.insert(lineSeg);
            }

		}//for-vcnt
	}//for-facetCnt

}//drawContour



