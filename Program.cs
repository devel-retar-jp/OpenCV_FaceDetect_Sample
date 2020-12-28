/// <summary>
/// OpenCV　による顔認識のサンプル
/// 
/// OpenCvSharpを使って認識させる方法
/// 画像処理のライブラリとして素晴らしい
/// 
/// 画像DB
/// http://www.imageprocessingplace.com/root_files_V3/image_databases.htm
/// 
///     2020/12/28  Retar.jp
/// 
/// </summary>
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

/// <summary>
/// 
/// </summary>
namespace OpenCVFaceDetect
{
    class Program
    {
        // 判定画像パス
        private const string INPUT_IMAGE_PATH = "lena_color_512.png";   //画像を用意してください
        private const string PADDING_IMAGE_PATH = "N225.png";           //画像を用意してください

        // 顔認識用カスケードファイルパス
        //https://github.com/opencv/opencv/tree/master/data/haarcascades
        private const string haarcascade_frontalface_eye = "haarcascade_eye.xml";
        private const string haarcascade_frontalface_alt = "haarcascade_frontalface_alt.xml";
        private const string haarcascade_frontalface_alt_tree = "haarcascade_frontalface_alt_tree.xml";
        private const string haarcascade_frontalface_alt2 = "haarcascade_frontalface_alt2.xml";
        private const string haarcascade_frontalface_default = "haarcascade_frontalface_default.xml";
        private const string haarcascade_smile = "haarcascade_smile.xml";
        private const string haarcascade_righteye_2splits = "haarcascade_righteye_2splits.xml";
        private const string haarcascade_lefteye_2splits = "haarcascade_lefteye_2splits.xml";

        //顔認識
        static Mat faces(CascadeClassifier ccf, Mat mat, Mat matRetImage, int R, int G, int B)
        {
            // 顔認識を実行
            var faces = ccf.DetectMultiScale(
                image: mat,
                scaleFactor: 1.08,
                minNeighbors: 2,
                flags: HaarDetectionType.ScaleImage,
                minSize: new Size(50, 50));

            using (var matSrcImage = new Mat(PADDING_IMAGE_PATH, ImreadModes.Color))
            using (var dst = new Mat())
            {
                // 認識した顔の周りを枠線で囲む
                foreach (var face in faces)
                {
                    //張り込み位置
                    var center = new Point
                    {
                        X = (int)(face.X ),                     
                        Y = (int)(face.Y + face.Height * 0.5)
                        //X = (int)(face.X - face.Width * 0.5),
                        //Y = (int)(face.Y - face.Height * 0.5)
                    };
                    //変形
                    var axes = new Size
                    {
                        Width = (int)(face.Width ),
                        //Height = (int)(face.Height )
                        //Width = (int)(face.Width * 0.5),
                        Height = (int)(face.Height * 0.5)
                    };
                    //変形
                    Cv2.Resize(matSrcImage, dst, axes, 0, 0, InterpolationFlags.Lanczos4);
                    // 画像の保存
                    Cv2.ImWrite(@"tmp.png", dst);
                    //重ねる
                    overlayImage(matRetImage, dst, center);
                    //●
                    //Cv2.Ellipse(matRetImage, center, axes, 0, 0, 360, new Scalar(R, G, B), 4);
                    //■
                    Cv2.Rectangle(
                        img: matRetImage,
                        rect: new Rect(face.X, face.Y, face.Width, face.Height),
                        color: new Scalar(R, G, B),
                        thickness: 2);
                }
            }
            return matRetImage;
        }
        
        //張り込み
        static Mat overlayImage(Mat src, Mat overlay, Point location)
        {
            //認識した位置に画像張り込み
            int ovfY = 0;
            for (int y = Math.Max(location.Y, 0); y < src.Rows; ++y)
            {
                int fY = y - location.Y;
                //fY = fY - (overlay.Rows/2);
                if (fY >= overlay.Rows)
                    break;
                int ovfX = 0;
                for (int x = Math.Max(location.X, 0); x < src.Cols; ++x)
                {
                    int fX = x - location.X;
                    if (fX >= overlay.Cols)
                        break;
                    /////////////
                    var pixel = src.Get<Vec3b>(fY, fX);
                    var pixelov = overlay.Get<Vec3b>(ovfY, ovfX);
                    double opacity = overlay.At<Vec4b>(ovfY, ovfX)[3] / 255;
                    var newPixel = new Vec3b
                    {
                        //画像貼り付け
                        Item0 = (byte)pixelov.Item0, // B
                        Item1 = (byte)pixelov.Item1, // G
                        Item2 = (byte)pixelov.Item2 // R
                        //反転貼り付け
                        //Item0 = (byte)(255 - pixelov.Item0), // B
                        //Item1 = (byte)(255 - pixelov.Item1), // G
                        //Item2 = (byte)(255 - pixelov.Item2) // R
                    };
                    //
                    src.Set<Vec3b>(y, x, newPixel);
                    ovfX++;
                }
                ovfY++;
            }
            return src;
        }

        static void Main(string[] args)
        {
            // 結果画像
            Mat matRetImage = null;

            // 顔認識用カスケード分類器を作成
            //using (var haarCascade_eye = new CascadeClassifier(haarcascade_frontalface_eye))
            using (var haarCascade_alt = new CascadeClassifier(haarcascade_frontalface_alt))
            //using (var haarCascade_alt_tree = new CascadeClassifier(haarcascade_frontalface_alt_tree))
            //using (var haarCascade_alt2 = new CascadeClassifier(haarcascade_frontalface_alt2))
            //using (var haarCascade_default = new CascadeClassifier(haarcascade_frontalface_default))
            //using (var haarCascade_smile = new CascadeClassifier(haarcascade_smile))
            //using (var haarCascade_righteye_2splits = new CascadeClassifier(haarcascade_righteye_2splits))
            //using (var haarCascade_lefteye_2splits = new CascadeClassifier(haarcascade_lefteye_2splits))
            // 判定画像ファイルをロード
            using (var matSrcImage = new Mat(INPUT_IMAGE_PATH, ImreadModes.Color))
            using (var matGrayscaleImage = new Mat())
            {
                matRetImage = matSrcImage.Clone();

                // 入力画像をグレースケール化
                Cv2.CvtColor(
                    src: matSrcImage,
                    dst: matGrayscaleImage,
                    code: ColorConversionCodes.BGR2GRAY);

                // 顔認識を実行
                //matRetImage = faces(haarCascade_eye, matGrayscaleImage, matRetImage, 255, 255, 0);
                matRetImage = faces(haarCascade_alt, matGrayscaleImage, matRetImage,255,0,0);
                //matRetImage = faces(haarCascade_alt_tree, matGrayscaleImage, matRetImage, 0, 255, 0);
                //matRetImage = faces(haarCascade_alt2, matGrayscaleImage, matRetImage, 0, 0, 255);
                //matRetImage = faces(haarCascade_default, matGrayscaleImage, matRetImage, 255, 255, 0);
                //matRetImage = faces(haarCascade_smile, matGrayscaleImage, matRetImage, 0, 255, 255);
                //matRetImage = faces(haarCascade_righteye_2splits, matGrayscaleImage, matRetImage, 255, 0, 0);
                //matRetImage = faces(haarCascade_lefteye_2splits, matGrayscaleImage, matRetImage, 0, 255, 0);

            }

            // 結果画像を表示
            Cv2.ImShow("Detected faces", matRetImage);

            // 画像の保存
            Cv2.ImWrite(@"output.png", matRetImage);

            // キー押下を待機
            Cv2.WaitKey();

            //解放
            matRetImage?.Dispose();

            //キー入力待ち
            //Console.ReadKey();
        }
    }
}
