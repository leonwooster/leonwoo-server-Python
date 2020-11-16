using System;
using System.IO;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataPath = @"Y:\Test_Projects\Python\course\opencv-dl-pytorch-week4\data\101\101_ObjectCategories\";

            string trainPath = @"Y:\Test_Projects\Python\course\opencv-dl-pytorch-week4\data\101\101_train\";
            string trainPathFiles = @"Y:\Test_Projects\Python\course\opencv-dl-pytorch-week4\data\101\train_paths.txt";

            string testPath = @"Y:\Test_Projects\Python\course\opencv-dl-pytorch-week4\data\101\101_test\";            
            string testPathFiles = @"Y:\Test_Projects\Python\course\opencv-dl-pytorch-week4\data\101\test_paths.txt";
                       
            try
            {
                CopyFiles(dataPath, trainPathFiles, trainPath);
                CopyFiles(dataPath, testPathFiles, testPath);
            }
            catch(Exception ex)
            {
                Console.WriteLine($"{ex.Message}");
            }

            Console.ReadLine();
        }

        static void CopyFiles(string dataPath, string files, string dest)
        {
            StreamReader file = new StreamReader(files);
            string line = "";
            string sourcePath = "";
            string destPath = "";
            string[] p = null;

            while ((line = file.ReadLine()) != null)
            {
                p = line.Split(new string[] { "/" }, StringSplitOptions.RemoveEmptyEntries);
                line = line.Replace("/", @"\");
                sourcePath = Path.Combine(dataPath, line);
                destPath = Path.Combine(dest, p[0]);

                if (!Directory.Exists(destPath))
                    Directory.CreateDirectory(destPath);

                destPath = Path.Combine(destPath, p[1]);

                Console.WriteLine($"Copying from {sourcePath} to {destPath}");
                File.Copy(sourcePath, destPath, true);
            }

            file.Close();
        }
    }
}
