using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using FFEA;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Xml.Serialization;

namespace Microsoft.Samples.Kinect.HDFaceBasics
{
    class SVMModel
    {
        balanceLibraries balanceLibraries = new balanceLibraries();

        /// <summary>
        /// function to return the database file number
        /// </summary>
        private int returnDataBaseFileNumber()
        {
            int fileNumber = 0;
            foreach (string file in Directory.EnumerateFiles(@"FFELibrary", "*.fp"))
            {
                fileNumber++;
            }
          
            return fileNumber;
        }

        /// <summary>
        /// build the SVM ML model for each selected AU 
        /// </summary>
        public List<Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear>> SVMForAllAUs()
        {
            List<Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear>> SVMlist = new List<Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear>>();

            double[][] inputs1 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs1 = new int[returnDataBaseFileNumber()];


            double[][] inputs2 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs2 = new int[returnDataBaseFileNumber()];

            double[][] inputs4 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs4 = new int[returnDataBaseFileNumber()];

            double[][] inputs5 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs5 = new int[returnDataBaseFileNumber()];

            double[][] inputs6 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs6 = new int[returnDataBaseFileNumber()];

            double[][] inputs10 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs10 = new int[returnDataBaseFileNumber()];

            double[][] inputs12 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs12 = new int[returnDataBaseFileNumber()];


            double[][] inputs15 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs15 = new int[returnDataBaseFileNumber()];

            double[][] inputs17 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs17 = new int[returnDataBaseFileNumber()];


            double[][] inputs26 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs26 = new int[returnDataBaseFileNumber()];

            double[][] inputs27 = new double[returnDataBaseFileNumber()][]; //88  arays in the database 
            int[] outputs27 = new int[returnDataBaseFileNumber()];

            int count = 0;



            foreach (string file in Directory.EnumerateFiles(@"FFELibrary", "*.fp"))

            {

                try
                {
                    inputs1[count] = new double[28];
                    inputs2[count] = new double[28];
                    inputs4[count] = new double[28];
                    inputs5[count] = new double[28];
                    inputs6[count] = new double[28];
                    inputs12[count] = new double[28];
                    inputs15[count] = new double[28];
                    inputs17[count] = new double[28];
                    inputs26[count] = new double[28];
                    inputs27[count] = new double[28];

                    XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                    TextReader reader = new StreamReader(file);
                    var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                    for (int m = 0; m < 28; m++)
                    {
                        //au1
                        if (m == 4 || m == 6 || m == 7 || m == 8 || m == 9 || m == 15 || m == 16 || m == 18 || m == 19 || m == 20)
                        {
                            inputs1[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs1[count] = 0;
                            else if (file.Contains("Laugh"))
                                outputs1[count] = 0;
                            else if (file.Contains("Shock"))
                                outputs1[count] = 1;
                            else if (file.Contains("Sad"))
                                outputs1[count] = 1;
                        }
                        //au2
                        if (m == 4 || m == 7 || m == 8 || m == 9 || m == 12)
                        {
                            inputs2[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs2[count] = 0;
                            else if (file.Contains("Laugh"))
                                outputs2[count] = 1;
                            else if (file.Contains("Shock"))
                                outputs2[count] = 1;
                            else if (file.Contains("Sad"))
                                outputs2[count] = 0;
                        }
                        //au4
                        if (m == 4 || m == 7 || m == 8 || m == 9 || m == 12 || m == 15 || m == 18 || m == 16 || m == 20 || m == 25)
                        {
                            inputs4[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs4[count] = 0;
                            else if (file.Contains("Laugh"))
                                outputs4[count] = 0;
                            else if (file.Contains("Shock"))
                                outputs4[count] = 0;
                            else if (file.Contains("Sad"))
                                outputs4[count] = 1;

                        }

                        //au5
                        if (m == 6 || m == 8 || m == 12 || m == 15 || m == 19 || m == 24 || m == 23 || m == 25)
                        {
                            inputs5[count][m] = read.Distances[m];
                            if (file.Contains("Smile"))
                                outputs5[count] = 0;
                            else if (file.Contains("Laugh"))
                                outputs5[count] = 1;
                            else if (file.Contains("Shock"))
                                outputs5[count] = 1;
                            else if (file.Contains("Sad"))
                                outputs5[count] = 0;
                        }
                        //au6
                        if (m == 10 || m == 11 || m == 13 || m == 21 || m == 23 || m == 24)
                        {
                            inputs6[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs6[count] = 1;
                            else if (file.Contains("Laugh"))
                                outputs6[count] = 1;
                            else if (file.Contains("Shock"))
                                outputs6[count] = 0;
                            else if (file.Contains("Sad"))
                                outputs6[count] = 0;
                        }

                        //12
                        if (m == 3 || m == 5 || m == 10 || m == 11 || m == 13 || m == 14 || m == 17 || m == 21)
                        {
                            inputs12[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs12[count] = 1;
                            else if (file.Contains("Laugh"))
                                outputs12[count] = 0;
                            else if (file.Contains("Shock"))
                                outputs12[count] = 0;
                            else if (file.Contains("Sad"))
                                outputs12[count] = 0;
                        }
                        //au15
                        if (m == 3 || m == 14 || m == 26 || m == 27)
                        {
                            inputs15[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs15[count] = 0;
                            else if (file.Contains("Laugh"))
                                outputs15[count] = 0;
                            else if (file.Contains("Shock"))
                                outputs15[count] = 0;
                            else if (file.Contains("Sad"))
                                outputs15[count] = 1;
                        }
                        //17

                        if (m == 0 || m == 1 || m == 22 || m == 3 || m == 14)
                        {
                            inputs17[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs17[count] = 0;
                            else if (file.Contains("Laugh"))
                                outputs17[count] = 0;
                            else if (file.Contains("Shock"))
                                outputs17[count] = 0;
                            else if (file.Contains("Sad"))
                                outputs17[count] = 1;
                        }


                        //26
                        if (m == 0 || m == 1 || m == 3 || m == 14 || m == 26 || m == 27)
                        {
                            inputs26[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs26[count] = 1;
                            else if (file.Contains("Laugh"))
                                outputs26[count] = 1;
                            else if (file.Contains("Shock"))
                                outputs26[count] = 0;
                            else if (file.Contains("Sad"))
                                outputs26[count] = 0;
                        }
                        //27
                        if (m == 0 || m == 1 || m == 3 || m == 10 || m == 14 || m == 26 || m == 27 || m == 21)
                        {
                            inputs27[count][m] = read.Distances[m];

                            if (file.Contains("Smile"))
                                outputs27[count] = 0;
                            else if (file.Contains("Laugh"))
                                outputs27[count] = 0;
                            else if (file.Contains("Shock"))
                                outputs27[count] = 1;
                            else if (file.Contains("Sad"))
                                outputs27[count] = 0;
                        }

                    }
                    reader.Close();

                    count++;

                }

                catch (Exception e)
                {
                    MessageBox.Show(e.ToString());
                }

            }

            var teacher1 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher1.ParallelOptions.MaxDegreeOfParallelism = 1;

            var teacher2 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher2.ParallelOptions.MaxDegreeOfParallelism = 1;
            var teacher3 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher3.ParallelOptions.MaxDegreeOfParallelism = 1;


            var teacher4 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher4.ParallelOptions.MaxDegreeOfParallelism = 1;

            var teacher6 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher6.ParallelOptions.MaxDegreeOfParallelism = 1;


            var teacher7 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher7.ParallelOptions.MaxDegreeOfParallelism = 1;


            var teacher15 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher15.ParallelOptions.MaxDegreeOfParallelism = 1;

            var teacher17 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher17.ParallelOptions.MaxDegreeOfParallelism = 1;

            var teacher26 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher26.ParallelOptions.MaxDegreeOfParallelism = 1;

            var teacher27 = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher27.ParallelOptions.MaxDegreeOfParallelism = 1;
            SVMlist.Add(teacher1.Learn(inputs1, outputs1)); //add au1 model
            SVMlist.Add(teacher2.Learn(inputs2, outputs2));//add au2 model
            SVMlist.Add(teacher3.Learn(inputs4, outputs4));//add au4 model
            SVMlist.Add(teacher4.Learn(inputs5, outputs5));//add au5 model
            SVMlist.Add(teacher6.Learn(inputs6, outputs6));//add au6 model
            SVMlist.Add(teacher7.Learn(inputs12, outputs12));//add au6 model
            SVMlist.Add(teacher15.Learn(inputs15, outputs15));//add au6 model
            SVMlist.Add(teacher17.Learn(inputs17, outputs17));//add au6 model
            SVMlist.Add(teacher26.Learn(inputs26, outputs26));//add au6 model
            SVMlist.Add(teacher27.Learn(inputs27, outputs27));//add au6 model

            return SVMlist;
        }


        /// <summary>
        /// .NET accord SVM classifier k-fold crossvalidation for AU1
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU1(List<string> AUs, int counts, int kfold, int kValue)
        {

            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            if (m == 4 || m == 6 || m == 7 || m == 8 || m == 9 || m == 15 || m == 16 || m == 18 || m == 19 || m == 20)
                                inputs[count][m] = read.Distances[m];

                        if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                            outputs[count] = 0;
                        else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                            outputs[count] = 0;
                        else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                            outputs[count] = 1;
                        else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                            outputs[count] = 1;


                        count++;

                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);

        }

        /// <summary>
        /// .NET accord svm classifier k-fold crossvalidation for AU2
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param> 
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU2(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            if (m == 4 || m == 7 || m == 8 || m == 9 || m == 12)
                            {
                                inputs[count][m] = read.Distances[m];
                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 0;
                            }


                        count++;

                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);

        }

        /// <summary>
        /// .NET accord svm classifier k-fold crossvalidation for AU4
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU4(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            if (m == 4 || m == 7 || m == 8 || m == 9 || m == 12 || m == 15 || m == 18 || m == 16 || m == 20 || m == 25)
                            {
                                inputs[count][m] = read.Distances[m];

                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 1;

                            }


                        count++;

                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);


        }


        /// <summary>
        /// .NET accord svm classifier k-fold crossvalidation for AU5
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU5(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            if (m == 6 || m == 8 || m == 12 || m == 15 || m == 19 || m == 24 || m == 23 || m == 25)
                            {
                                inputs[count][m] = read.Distances[m];
                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 0;
                            }


                        count++;

                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);
        }


        /// <summary>
        /// .NET accord svm classifier k-fold crossvalidation for AU6
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU6(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            //au6
                            if (m == 10 || m == 11 || m == 13 || m == 21 || m == 23 || m == 24)
                            {
                                inputs[count][m] = read.Distances[m];

                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 0;
                            }

                        count++;

                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);

        }


        /// <summary>
        /// .NET accord svm classifier k-fold crossvalidation for AU12
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU12(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            //12
                            if (m == 3 || m == 5 || m == 10 || m == 11 || m == 13 || m == 14 || m == 17 || m == 21)
                            {
                                inputs[count][m] = read.Distances[m];

                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 0;
                            }


                        count++;

                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);
        }

        /// <summary>
        /// .NET accord svm classifier k-fold crossvalidation for AU15
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU15(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            //12
                            if (m == 3 || m == 14 || m == 26 || m == 27)
                            {
                                inputs[count][m] = read.Distances[m];

                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 1;
                            }

                        count++;
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);

        }


        /// <summary>
        /// .NET accord svm classifier k-fold crossvalidation for AU17
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU17(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            //17
                            if (m == 0 || m == 1 || m == 22 || m == 3 || m == 14)
                            {
                                inputs[count][m] = read.Distances[m];

                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 1;
                            }

                        count++;
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);

        }


        /// <summary>
        /// .NET accord svm classifier k-fold crossvalidation for AU26
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU26(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            //26
                            if (m == 0 || m == 1 || m == 3 || m == 14 || m == 26 || m == 27)
                            {
                                inputs[count][m] = read.Distances[m];

                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 0;
                            }

                        count++;
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);

        }

        /// <summary>
        /// .NET accord svm classifier for k-fold crossvalidation ofr AU27
        /// </summary>
        /// <param name="AUs_List"></param>
        /// <param name="counts"></param>
        /// <param name="kfold"></param>
        /// <param name="kValue"></param>
        public Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> crossValidateAU27(List<string> AUs, int counts, int kfold, int kValue)
        {
            int count = 0;
            double[][] inputs = new double[balanceLibraries.balanceLibrary(AUs).Count() - kfold][]; //88  arays in the database 
            int[] outputs = new int[balanceLibraries.balanceLibrary(AUs).Count() - kfold];

            for (int i = 0; i < balanceLibraries.balanceLibrary(AUs).Count(); i++)
            {
                XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                if (i < (counts - 1) * kfold || i >= counts * kfold)
                {
                    TextReader reader = new StreamReader(balanceLibraries.balanceLibrary(AUs)[i]);
                    try
                    {
                        inputs[count] = new double[28];
                        var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                        for (int m = 0; m < 28; m++)

                            //27
                            if (m == 0 || m == 1 || m == 3 || m == 10 || m == 14 || m == 26 || m == 27 || m == 21)
                            {
                                inputs[count][m] = read.Distances[m];

                                if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Smile"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Laugh"))
                                    outputs[count] = 0;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Shock"))
                                    outputs[count] = 1;
                                else if (balanceLibraries.balanceLibrary(AUs)[i].Contains("Sad"))
                                    outputs[count] = 0;
                            }


                        count++;
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(ex.ToString());
                    }
                    reader.Close();
                }
            }
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            // Configure parallel execution options
            teacher.ParallelOptions.MaxDegreeOfParallelism = 1;
            return teacher.Learn(inputs, outputs);

        }
    }
}
