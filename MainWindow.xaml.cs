
namespace FFEA
{
    using System;
    using System.Windows;
    using System.Windows.Media;
    using System.Windows.Media.Media3D;
    using Microsoft.Kinect;
    using Microsoft.Kinect.Face;
    using System.IO;
    using System.Collections.Generic;
    using System.Threading;
    using System.Xml.Serialization;
    using System.Windows.Media.Imaging;
    using System.Linq;
    using Accord.Statistics.Kernels;
    using Microsoft.Samples.Kinect.HDFaceBasics;
    using System.Windows.Controls;
    using System.Windows.Data;


    /// <summary>
    /// Main Window
    /// </summary>
    public partial class MainWindow : Window, IDisposable
    {
        /// <summary>
        /// Currently used KinectSensor
        /// </summary>
        private KinectSensor sensor = null;

        /// <summary>
        /// rgb reader
        /// </summary>
        WriteableBitmap ColorBitmapReader = null;

        /// <summary>
        /// set the recording status
        /// </summary>
        public bool IsRecording = false;


        public FaceFeature TheFaceWriter = new FaceFeature();
        public FaceFeature TheFaceReader = new FaceFeature();

        /// <summary>
        /// Body frame source to get a BodyFrameReader
        /// </summary>
        private BodyFrameSource bodySource = null;

        /// <summary>
        /// Body frame reader to get body frames
        /// </summary>
        private BodyFrameReader bodyReader = null;

        /// <summary>
        /// HighDefinitionFaceFrameSource to get a reader and a builder from.
        /// Also to set the currently tracked user id to get High Definition Face Frames of
        /// </summary>
        private HighDefinitionFaceFrameSource highDefinitionFaceFrameSource = null;

        /// <summary>
        /// HighDefinitionFaceFrameReader to read HighDefinitionFaceFrame to get FaceAlignment
        /// </summary>
        private HighDefinitionFaceFrameReader highDefinitionFaceFrameReader = null;

        /// <summary>
        /// FaceAlignment is the result of tracking a face, it has face animations location and orientation
        /// </summary>
        private FaceAlignment currentFaceAlignment = null;

        /// <summary>
        /// FaceModel is a result of capturing a face
        /// </summary>
        private FaceModel currentFaceModel = null;

        /// <summary>
        /// FaceModelBuilder is used to produce a FaceModel
        /// </summary>
        private FaceModelBuilder faceModelBuilder = null;

        /// <summary>
        /// The currently tracked body
        /// </summary>
        private Body currentTrackedBody = null;

        /// <summary>
        /// The currently tracked body
        /// </summary>
        private ulong currentTrackingId = 0;

        /// <summary>
        /// Gets or sets the current tracked user id
        /// </summary>
        private string currentBuilderStatus = string.Empty;

        /// <summary>
        /// Initializes a new instance of the MainWindow class.
        /// </summary>
        private readonly int BytePerPixel = (PixelFormats.Bgra32.BitsPerPixel + 7) / 8;

        private ColorFrameReader ColorReader = null;

        /// <summary>
        /// Kinect colour frame
        /// </summary>
        private byte[] ColorPixels = null;

        /// <summary>
        /// Kinect depth channel
        /// </summary>
        private WriteableBitmap ColorBitmap = null;

        public Dictionary<string, List<int>> ActivatedAUs { get; set; }

        //initialise intances of each knn classifier for au
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC1;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC2;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC4;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC5;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC6;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC12;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC15;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC17;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC26;
        static private Accord.MachineLearning.KNearestNeighbors<double[]> knnC27;

        //initialise intances of each svm classifier for au
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm1;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm2;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm4;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm5;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm6;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm12;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm15;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm17;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm26;
        static private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> svm27;


        private Accord.MachineLearning.KNearestNeighbors<double[]> KnnAUCossvalidation;
        private Accord.MachineLearning.VectorMachines.MulticlassSupportVectorMachine<Linear> SvmAUCossvalidation;

        KNNModel knnModel = new KNNModel();
        SVMModel svmModel = new SVMModel();

        balanceLibraries balancedLibrary = new balanceLibraries();

        public MainWindow()
        {
            this.InitializeComponent();
            this.InitializeHDFace();
            this.InitializeCamera();
            this.DataContext = this;

            //instantiate the trained svm model upon launch application
            svm1 = svmModel.SVMForAllAUs()[0];
            svm2 = svmModel.SVMForAllAUs()[1];
            svm4 = svmModel.SVMForAllAUs()[2];
            svm5 = svmModel.SVMForAllAUs()[3];
            svm6 = svmModel.SVMForAllAUs()[4];
            svm12 = svmModel.SVMForAllAUs()[5];
            svm15 = svmModel.SVMForAllAUs()[6];
            svm17 = svmModel.SVMForAllAUs()[7];
            svm26 = svmModel.SVMForAllAUs()[8];
            svm27 = svmModel.SVMForAllAUs()[9];


            //give the activated au list a header from au1 to 27
            List<int> sample = new List<int>();
            for (int k = 0; k < aus().Count; k++)
            {
                DataGridTextColumn cols = new DataGridTextColumn();

                cols.Header = "AU" + aus()[k];
                cols.Binding = new Binding(string.Format("Value[{0}]", k));
                dataGridAU.Columns.Add(cols);
            }
        }

        /// <summary>
        /// initialize the rgb camera of kinect 
        /// </summary>
        private void InitializeCamera()
        {
            FrameDescription desc = sensor.ColorFrameSource.FrameDescription;

            ColorReader = sensor.ColorFrameSource.OpenReader();
            ColorPixels = new byte[desc.Width * desc.Height * BytePerPixel];
            ColorBitmap = new WriteableBitmap(desc.Width, desc.Height, -1, -1, PixelFormats.Bgra32, null);

            ColorReader.FrameArrived += ColorFrame_Arrived;
        }

        /// <summary>
        /// formation of the RGB frame
        /// </summary>
        private void ColorFrame_Arrived(object sender, ColorFrameArrivedEventArgs e)
        {
            ColorFrame frame = e.FrameReference.AcquireFrame();
            if (frame == null) return;

            using (frame)
            {
                FrameDescription frameDesc = frame.FrameDescription;

                frame.CopyConvertedFrameDataToArray(ColorPixels, ColorImageFormat.Bgra);
                ColorBitmap.WritePixels(new Int32Rect(0, 0, frameDesc.Width, frameDesc.Height), ColorPixels, frameDesc.Width * BytePerPixel, 0);
            }
        }

        /// <summary>
        /// Gets or sets the current tracked user id
        /// </summary>
        private ulong CurrentTrackingId
        {
            get
            {
                return this.currentTrackingId;
            }

            set
            {
                this.currentTrackingId = value;
            }
        }


        /// <summary>
        /// Gets or sets the current Face Builder instructions to user
        /// </summary>
        private string CurrentBuilderStatus
        {
            get
            {
                return this.currentBuilderStatus;
            }

            set
            {
                this.currentBuilderStatus = value;
            }
        }

        /// <summary>
        /// Called when disposed of
        /// </summary>
        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Dispose based on whether or not managed or native resources should be freed
        /// </summary>
        /// <param name="disposing">Set to true to free both native and managed resources, false otherwise</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (this.currentFaceModel != null)
                {
                    this.currentFaceModel.Dispose();
                    this.currentFaceModel = null;
                }
            }
        }

        /// <summary>
        /// Returns the length of a vector from origin
        /// </summary>
        /// <param name="point">Point in space to find it's distance from origin</param>
        /// <returns>Distance from origin</returns>
        private static double VectorLength(CameraSpacePoint point)
        {
            var result = Math.Pow(point.X, 2) + Math.Pow(point.Y, 2) + Math.Pow(point.Z, 2);

            result = Math.Sqrt(result);

            return result;
        }

        /// <summary>
        /// Finds the closest body from the sensor if any
        /// </summary>
        /// <param name="bodyFrame">A body frame</param>
        /// <returns>Closest body, null of none</returns>
        private static Body FindClosestBody(BodyFrame bodyFrame)
        {
            Body result = null;
            double closestBodyDistance = double.MaxValue;

            Body[] bodies = new Body[bodyFrame.BodyCount];
            bodyFrame.GetAndRefreshBodyData(bodies);

            foreach (var body in bodies)
            {
                if (body.IsTracked)
                {
                    var currentLocation = body.Joints[JointType.SpineBase].Position;

                    var currentDistance = VectorLength(currentLocation);

                    if (result == null || currentDistance < closestBodyDistance)
                    {
                        result = body;
                        closestBodyDistance = currentDistance;
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Find if there is a body tracked with the given trackingId
        /// </summary>
        /// <returns>The body object, null of none</returns>
        private static Body FindBodyWithTrackingId(BodyFrame bodyFrame, ulong trackingId)
        {
            Body result = null;

            Body[] bodies = new Body[bodyFrame.BodyCount];
            bodyFrame.GetAndRefreshBodyData(bodies);

            foreach (var body in bodies)
            {
                if (body.IsTracked)
                {
                    if (body.TrackingId == trackingId)
                    {
                        result = body;
                        break;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Fires when Window is Loaded
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            this.InitializeHDFace();
        }

        /// <summary>
        /// Initialize Kinect object
        /// </summary>
        private void InitializeHDFace()
        {
            this.sensor = KinectSensor.GetDefault();
            this.bodySource = this.sensor.BodyFrameSource;
            this.bodyReader = this.bodySource.OpenReader();
            this.bodyReader.FrameArrived += this.BodyReader_FrameArrived;

            this.highDefinitionFaceFrameSource = new HighDefinitionFaceFrameSource(this.sensor);
            this.highDefinitionFaceFrameSource.TrackingIdLost += this.HdFaceSource_TrackingIdLost;

            this.highDefinitionFaceFrameReader = this.highDefinitionFaceFrameSource.OpenReader();
            this.highDefinitionFaceFrameReader.FrameArrived += this.HdFaceReader_FrameArrived;

            this.currentFaceModel = new FaceModel();
            this.currentFaceAlignment = new FaceAlignment();

            this.InitializeMesh();
            // this.UpdateMesh();
            upGetMesh();
            this.sensor.Open();
        }

        /// <summary>
        /// Initializes a 3D mesh to deform every frame
        /// </summary>
        private void InitializeMesh()
        {
            var vertices = this.currentFaceModel.CalculateVerticesForAlignment(this.currentFaceAlignment);

            var triangleIndices = this.currentFaceModel.TriangleIndices;

            var indices = new Int32Collection(triangleIndices.Count);

            for (int i = 0; i < triangleIndices.Count; i += 3)
            {

                uint index01 = triangleIndices[i];
                uint index02 = triangleIndices[i + 1];
                uint index03 = triangleIndices[i + 2];

                indices.Add((int)index03);
                indices.Add((int)index02);
                indices.Add((int)index01);
            }

            this.theGeometry.TriangleIndices = indices;
            this.theGeometry.Normals = null;
            this.theGeometry.Positions = new Point3DCollection();
            this.theGeometry.TextureCoordinates = new PointCollection();

            foreach (var vert in vertices)
            {
                this.theGeometry.Positions.Add(new Point3D(vert.X, vert.Y, -vert.Z));
                this.theGeometry.TextureCoordinates.Add(new Point());
            }

        }
        /// <summary>
        /// This event is fired when a new HDFace frame is ready for consumption
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void HdFaceReader_FrameArrived(object sender, HighDefinitionFaceFrameArrivedEventArgs e)
        {
            using (var frame = e.FrameReference.AcquireFrame())
            {
                // We might miss the chance to acquire the frame; it will be null if it's missed.
                // Also ignore this frame if face tracking failed.
                if (frame == null || !frame.IsFaceTracked)
                {
                    return;
                }

                frame.GetAndRefreshFaceAlignmentResult(this.currentFaceAlignment);
                // this.UpdateMesh();
                upGetMesh();

                if (IsRecording)
                    Recording();
            }

        }


        /// <summary>
        /// This event is fired when a tracking is lost for a body tracked by HDFace Tracker
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void HdFaceSource_TrackingIdLost(object sender, TrackingIdLostEventArgs e)
        {
            var lostTrackingID = e.TrackingId;

            if (this.CurrentTrackingId == lostTrackingID)
            {
                this.CurrentTrackingId = 0;
                this.currentTrackedBody = null;
                if (this.faceModelBuilder != null)
                {
                    this.faceModelBuilder.Dispose();
                    this.faceModelBuilder = null;
                }

                this.highDefinitionFaceFrameSource.TrackingId = 0;
            }
        }

        /// <summary>
        /// Initialize the colour footage reader
        /// </summary>
        private void InitializeHDFaceReader()
        {
            this.currentFaceModel = new FaceModel();
            this.currentFaceAlignment = new FaceAlignment();
            this.InitializeMesh();
            FaceIndex = 0;
        }

        /// <summary>
        /// read the 3D mesh of loaded footage 
        /// </summary>
        private void ReadMesh()
        {
            FaceFrame CurrentFace = new FaceFrame(TheFaceReader.DepthFootage[FaceIndex].Frame);

            for (int i = 0; i < 1347; i++)
            {
                var vert = CurrentFace.Frame[i];
                //devide by 6.5 to rescale the data to fit the geometry model
                this.theGeometry.Positions[i] = new Point3D((vert.X / 6.5), (vert.Y / 6.5), -(vert.Z / 6.5));
            }

        }

        /// <summary>
        /// This event fires when a BodyFrame is ready for consumption
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void BodyReader_FrameArrived(object sender, BodyFrameArrivedEventArgs e)
        {
            var frameReference = e.FrameReference;
            using (var frame = frameReference.AcquireFrame())
            {
                if (frame == null)
                {
                    // We might miss the chance to acquire the frame, it will be null if it's missed
                    return;
                }

                if (this.currentTrackedBody != null)
                {
                    this.currentTrackedBody = FindBodyWithTrackingId(frame, this.CurrentTrackingId);

                    if (this.currentTrackedBody != null)
                    {
                        return;
                    }
                }

                Body selectedBody = FindClosestBody(frame);

                if (selectedBody == null)
                {
                    return;
                }

                this.currentTrackedBody = selectedBody;
                this.CurrentTrackingId = selectedBody.TrackingId;

                this.highDefinitionFaceFrameSource.TrackingId = this.CurrentTrackingId;
            }
        }


        /// <summary>
        /// record the captured depth and RGB face model 
        /// </summary>
        public void Recording()
        {
            {
                var vertices = this.currentFaceModel.CalculateVerticesForAlignment(this.currentFaceAlignment);
                Point3D[] TempPoint3D = new Point3D[1347];

                for (int i = 0; i < vertices.Count; i++)
                {
                    var vert = vertices[i];
                    TempPoint3D[i] = new Point3D(vert.X, vert.Y, -vert.Z);
                }

                TheFaceWriter.DepthFootage.Add(new FaceFrame(TempPoint3D));
                byte[] RGBWriter = new byte[280 * 320 * BytePerPixel];

                //crop image to minimise the data size
                ColorBitmap.CopyPixels(new Int32Rect(870, 240, 280, 320), RGBWriter, 280 * BytePerPixel, 0);

                TheFaceWriter.RGBFootage.Add(RGBWriter);
                TheFaceWriter.IsExpression.Add(false);
            }
        }


        /// <summary>
        /// Sends the new deformed mesh to be drawn
        /// </summary>
        private void upGetMesh()
        {
            var vertices = this.currentFaceModel.CalculateVerticesForAlignment(this.currentFaceAlignment);
            Point3D[] TempPoint3D = new Point3D[1347];

            for (int i = 0; i < vertices.Count; i++) //loop all the vertices in the face model
            {
                var vert = vertices[i]; //assigned vertice by  index
                Point3D temp = new Point3D(vert.X, vert.Y, -vert.Z);
                this.theGeometry.Positions[i] = temp;
            }

        }


        /// <summary>
        /// review the RGB load kinect footage
        /// </summary>
        private void ReadColorFrame()
        {
            byte[] RGBReader = TheFaceReader.RGBFootage[FaceIndex++];
            ColorBitmapReader.WritePixels(new Int32Rect(0, 0, 280, 320), RGBReader, 280 * BytePerPixel, 0);
            reviewFootage.Source = ColorBitmapReader;
        }

        /// <summary>
        /// Start a face capture operation
        /// </summary>
        private void StartCapture()
        {
            this.StopFaceCapture();

            this.faceModelBuilder = null;

            this.faceModelBuilder = this.highDefinitionFaceFrameSource.OpenModelBuilder(FaceModelBuilderAttributes.None);

            this.faceModelBuilder.BeginFaceDataCollection();

            this.faceModelBuilder.CollectionCompleted += this.HdFaceBuilder_CollectionCompleted;
        }

        /// <summary>
        /// Cancel the current face capture operation
        /// </summary>
        private void StopFaceCapture()
        {
            if (this.faceModelBuilder != null)
            {
                this.faceModelBuilder.Dispose();
                this.faceModelBuilder = null;
            }
        }

        /// <summary>
        /// This event fires when the face capture operation is completed
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void HdFaceBuilder_CollectionCompleted(object sender, FaceModelBuilderCollectionCompletedEventArgs e)
        {
            var modelData = e.ModelData;

            this.currentFaceModel = modelData.ProduceFaceModel();

            this.faceModelBuilder.Dispose();
            this.faceModelBuilder = null;
        }

        /////////////////////////////////
        /// functions for Tab page1 face
        /////////////////////////////////


        /// <summary>
        /// event fired when button is clicked for recording footage
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void Button_Record_Click(object sender, RoutedEventArgs e)
        {
            IsRecording = !IsRecording;

            if (IsRecording)
            {
                TheFaceWriter = new FaceFeature();
                Button_Recording.Background = Brushes.Green;
                Button_Recording.Content = "Recording";
            }
            else
            {
                Button_Recording.Background = Brushes.Red;
                Button_Recording.Content = "Recording Stopped";
            }
        }


        /// <summary>
        /// save the recorded footage into local file
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void SaveFootage_Click(object sender, RoutedEventArgs e)
        {
            bool isSucceed = true;

            try { TheFaceWriter.ID = Convert.ToInt32(IDtextBox.Text); }
            catch (System.FormatException)
            {
                MessageBox.Show("ID is not a valid number");
                isSucceed = false;
            }

            try { TheFaceWriter.Age = Convert.ToInt32(AgeTextBox.Text); }
            catch (System.FormatException)
            {
                MessageBox.Show("Age is not a valid number");
                isSucceed = false;
            }

            TheFaceWriter.IsMale = (bool)MaleButton.IsChecked;


            try { TheFaceWriter.ExpressionType = ((ComboBoxItem)ExpressioncomboBox.SelectedItem).Content.ToString(); }
            catch (System.NullReferenceException)
            {
                TheFaceWriter.ExpressionType = "";
            }

            TheFaceWriter.Ethnicity = RaceTextBox.Text;


            if (isSucceed)
            {
                Microsoft.Win32.SaveFileDialog dlg = new Microsoft.Win32.SaveFileDialog();
                string documentName = TheFaceWriter.ID + "_" + TheFaceWriter.ExpressionType;
                dlg.FileName = documentName;
                dlg.DefaultExt = ".kf"; // Default file extension
                dlg.Filter = "Kenect face data|*.kf"; // Filter files by extension

                // Show save file dialog box
                Nullable<bool> result = dlg.ShowDialog();

                // Process save file dialog box results
                if (result == true)
                {
                    // Save document
                    MessageBox.Show(dlg.FileName);
                    XmlSerializer serializer = new XmlSerializer(typeof(FaceFeature));
                    TextWriter writer = new StreamWriter(dlg.FileName);
                    serializer.Serialize(writer, TheFaceWriter);
                    writer.Close();
                }
            }
        }

        /// <summary>
        /// load footage for playing 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void loadFootage_Click(object sender, RoutedEventArgs e)
        {
            featureExtraction f = new featureExtraction();
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();
            dlg.DefaultExt = ".kf"; // Default file extension
            dlg.Filter = "Kinect face data|*.kf"; // Filter files by extension

            // Show read file dialog box
            Nullable<bool> result = dlg.ShowDialog();

            // Process read file dialog box results
            if (result == true)
            {
                bool isSucceed = true;

                // read document
                XmlSerializer deserializer = new XmlSerializer(typeof(FaceFeature));
                TextReader reader = new StreamReader(dlg.FileName);
                try { TheFaceReader = (FaceFeature)deserializer.Deserialize(reader); }
                catch (System.InvalidOperationException)
                {
                    isSucceed = false;
                    MessageBox.Show(dlg.FileName + " is not a vaild file");
                }
                reader.Close();

                if (isSucceed)
                {
                    InitializeHDFaceReader();
                    upGetMesh();
                    ColorBitmapReader = new WriteableBitmap(280, 320, -1, -1, PixelFormats.Bgra32, null);
                    ReadColorFrame();
                    loadedFileName.Content = dlg.FileName;
                    MessageBox.Show("loaded");
                    playForward.IsEnabled = true;
                    Savepattern.IsEnabled = true;
                    playBack.IsEnabled = true;
                }
            }
        }

        private int FaceIndex;

        /// <summary>
        /// play forward the footage
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void playFootage_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                upGetMesh();
                ReadMesh();
                ReadColorFrame();
            }
            catch (System.ArgumentOutOfRangeException)
            {
                FaceIndex++;
                MessageBox.Show("The footage has reached end");
            }
        }

        /// <summary>
        /// back play the loaded footage
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void PlayBack_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                upGetMesh();
                ReadMesh();
                ReadColorFrame();

            }
            catch (System.ArgumentOutOfRangeException)
            {
                FaceIndex--;
                MessageBox.Show("The footage has reached start");

            }
        }

        /// <summary>
        /// extract the pattern from the loaded kinect footage
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void saveAsPattern_clicked(object sender, RoutedEventArgs e)
        {
            getDsitanceToNoseTipInOneFrame framesaver = TheFaceReader.averageDistance();
            Microsoft.Win32.SaveFileDialog dlg = new Microsoft.Win32.SaveFileDialog();
            string DocumentName = TheFaceReader.ID + "_" + TheFaceReader.ExpressionType;
            dlg.FileName = DocumentName;
            dlg.DefaultExt = ".fp"; // Default file extension
            dlg.Filter = "Face pattern data|*.fp"; // Filter files by extension
            Nullable<bool> result = dlg.ShowDialog();
            try
            {
                if (result == true)
                {
                    XmlSerializer serializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                    TextWriter writer = new StreamWriter(dlg.FileName);
                    serializer.Serialize(writer, framesaver);
                    MessageBox.Show("Done");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }
        }



        /////////////////////////////////////////
        /// functions for Tab page2 data analysis
        /////////////////////////////////////////


        /// <summary>
        /// load a footage and review its infomation about the pattern and frames points
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void analysis_clicked(object sender, RoutedEventArgs e)
        {
            featureExtraction fe = new featureExtraction();

            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();
            dlg.DefaultExt = ".kf"; // Default file extension
            dlg.Filter = "Kinect Face data |*.kf"; // Filter files by extension

            Nullable<bool> result = dlg.ShowDialog();
            Expression getdistance = new Expression();

            if (result == true)
            {
                // Read document
                XmlSerializer deserializer = new XmlSerializer(typeof(FaceFeature));
                TextReader reader = new StreamReader(dlg.FileName);
                //fe.expressions.Content = dlg.FileName;
                fe.docName.Content = dlg.FileName;

                getdistance = ((FaceFeature)deserializer.Deserialize(reader)).expression();

                // gc = new GridViewColumn();
                for (int k = 1; k < getdistance.expression.Count(); k++)
                {
                    List<double> n = new List<double>();

                    for (int i = 0; i < Enum.GetValues(typeof(FeaturePoints)).Length; i++)
                    {
                        n.Add(Math.Round(Convert.ToDouble(getdistance.expression[k].Distances[i]), 9, MidpointRounding.AwayFromZero)); //depth must start from 1   

                    }
                    fe.getOneFootage.Add(k, n);
                }
                fe.Show();

                reader.Close();
            }

        }


        /// <summary>
        /// refresh the library display them all in box
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void refresh_library_clicked(object sender, RoutedEventArgs e)
        {
            TextBox_Library.Text = "";

            foreach (string file in Directory.EnumerateFiles(@"FFELibrary", "*.fp"))

            {
                TextBox_Library.Text += file + '\n';

            }
            database.IsEnabled = true;
        }


        /// <summary>
        /// This event will trigger when the user click the Satistic button. It will plot and show the satistical result of the database in a new window.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Database_Click(object sender, RoutedEventArgs e)
        {
            Dictionary<string, int> SatisticalData = new Dictionary<string, int>();


            foreach (string file in Directory.EnumerateFiles(@"FFELibrary", "*.fp"))
            {
                if (file.Contains("Smile"))
                {
                    if (SatisticalData.Keys.Contains("Smile"))
                        SatisticalData["Smile"]++;
                    else
                        SatisticalData.Add("Smile", 1);
                }
                if (file.Contains("Sad"))
                {
                    if (SatisticalData.Keys.Contains("Sad"))
                        SatisticalData["Sad"]++;
                    else
                        SatisticalData.Add("Sad", 1);
                }
                if (file.Contains("Shock"))
                {
                    if (SatisticalData.Keys.Contains("Shock"))
                        SatisticalData["Shock"]++;
                    else
                        SatisticalData.Add("Shock", 1);
                }
                if (file.Contains("Laugh"))
                {
                    if (SatisticalData.Keys.Contains("Laugh"))
                        SatisticalData["Laugh"]++;
                    else
                        SatisticalData.Add("Laugh", 1);
                }

            }

            Statistic ViewSatistic = new Statistic(SatisticalData);
            ViewSatistic.ShowDialog();
        }


        /// <summary>
        /// .NET AUS k-fold cross validation 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void AUCrossValidate_Clicked(object sender, RoutedEventArgs e)
        {
            croseValidation.Text = "";
            int kValue = Convert.ToInt32(((ComboBoxItem)KvalueBox.SelectedItem).Content);
            double AU1YY = 0;
            double AU1YN = 0;
            double AU1NY = 0;
            double AU1NN = 0;
            int kfold = 4;
            int countfold = 1;
            List<string> corssValidateAU1 = new List<string>();
            corssValidateAU1.Add("Shock");
            corssValidateAU1.Add("Sad");
            double AU2YY = 0;
            double AU2YN = 0;
            double AU2NY = 0;
            double AU2NN = 0;
            int kfoldAU2 = 4;
            int countfoldAU2 = 1;
            List<string> corssValidationAU2 = new List<string>();
            corssValidationAU2.Add("Shock");
            corssValidationAU2.Add("Laugh");
            double AU4YY = 0;
            double AU4YN = 0;
            double AU4NY = 0;
            double AU4NN = 0;
            int kfoldAU4 = 4;
            int countfoldAU4 = 1;
            List<string> corssValidationAU4 = new List<string>();
            corssValidationAU4.Add("Sad");
            double AU5YY = 0;
            double AU5YN = 0;
            double AU5NY = 0;
            double AU5NN = 0;
            int kfoldAU5 = 4;
            int countfoldAU5 = 1;
            List<string> corssValidationAU5 = new List<string>();
            corssValidationAU5.Add("Laugh");
            corssValidationAU5.Add("Shock");
            double AU6YY = 0;
            double AU6YN = 0;
            double AU6NY = 0;
            double AU6NN = 0;
            int kfoldAU6 = 4;
            int countfoldAU6 = 1;
            List<string> corssValidationAU6 = new List<string>();
            corssValidationAU6.Add("Smile");
            corssValidationAU6.Add("Laugh");
            double AU12YY = 0;
            double AU12YN = 0;
            double AU12NY = 0;
            double AU12NN = 0;
            int kfoldAU12 = 4;
            int countfoldAU12 = 1;
            List<string> corssValidationAU12 = new List<string>();
            corssValidationAU12.Add("Smile");
            double AU15YY = 0;
            double AU15YN = 0;
            double AU15NY = 0;
            double AU15NN = 0;
            int kfoldAU15 = 4;
            int countfoldAU15 = 1;
            List<string> corssValidationAU15 = new List<string>();
            corssValidationAU15.Add("Sad");
            double AU17YY = 0;
            double AU17YN = 0;
            double AU17NY = 0;
            double AU17NN = 0;
            int kfoldAU17 = 4;
            int countfoldAU17 = 1;
            List<string> corssValidationAU17 = new List<string>();
            corssValidationAU17.Add("Sad");
            double AU26YY = 0;
            double AU26YN = 0;
            double AU26NY = 0;
            double AU26NN = 0;
            int kfoldAU26 = 4;
            int countfoldAU26 = 1;
            List<string> corssValidationAU26 = new List<string>();
            corssValidationAU26.Add("Smile");
            corssValidationAU26.Add("Laugh");
            double AU27YY = 0;
            double AU27YN = 0;
            double AU27NY = 0;
            double AU27NN = 0;
            int kfoldAU27 = 4;
            int countfoldAU27 = 1;
            List<string> corssValidationAU27 = new List<string>();
            corssValidationAU27.Add("Shock");
            if (checkKNN.IsChecked == true)
            {//using knn
                ThreadPool.QueueUserWorkItem((o) =>
                {
                    Dispatcher.Invoke((Action)(() =>
                    {
                        CrossValidateAU.IsEnabled = false;
                    }));

                    while (countfold <= balancedLibrary.balanceLibrary(corssValidateAU1).Count() / kfold)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 10 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU1(corssValidateAU1, countfold, kfoldAU2, kValue);


                        for (int i = kfold * (countfold - 1); i < kfold * countfold; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidateAU1)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Shock"))
                                {
                                    AU1YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Sad"))
                                {
                                    AU1YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Laugh"))
                                {
                                    AU1NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Smile"))
                                {
                                    AU1NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Shock"))
                                {
                                    AU1YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Sad"))
                                {
                                    AU1YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Laugh"))
                                {
                                    AU1NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Smile"))
                                {
                                    AU1NN++;
                                }
                            }
                        }
                        countfold++;

                    }

                    //for au2
                    while (countfoldAU2 <= balancedLibrary.balanceLibrary(corssValidationAU2).Count() / kfoldAU2)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 20 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU2(corssValidationAU2, countfoldAU2, kfoldAU2, kValue);

                        for (int i = kfoldAU2 * (countfoldAU2 - 1); i < kfoldAU2 * countfoldAU2; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU2)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Shock"))
                                {
                                    AU2YY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Sad"))
                                {
                                    AU2NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Laugh"))
                                {
                                    AU2YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Smile"))
                                {
                                    AU2NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Shock"))
                                {
                                    AU2YN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Sad"))
                                {
                                    AU2NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Laugh"))
                                {
                                    AU2YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Smile"))
                                {
                                    AU2NN++;
                                }
                            }
                        }
                        countfoldAU2++;
                    }

                    //for au4
                    while (countfoldAU4 <= balancedLibrary.balanceLibrary(corssValidationAU4).Count() / kfoldAU4)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 30 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU4(corssValidationAU4, countfoldAU4, kfoldAU4, kValue);

                        for (int i = kfoldAU4 * (countfoldAU4 - 1); i < kfoldAU4 * countfoldAU4; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU4)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Shock"))
                                {
                                    AU4NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Sad"))
                                {
                                    AU4YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Laugh"))
                                {
                                    AU4NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Smile"))
                                {
                                    AU4NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Shock"))
                                {
                                    AU4NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Sad"))
                                {
                                    AU4YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Laugh"))
                                {
                                    AU4NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Smile"))
                                {
                                    AU4NN++;
                                }
                            }
                        }
                        countfoldAU4++;
                    }

                    //for au5
                    while (countfoldAU5 <= balancedLibrary.balanceLibrary(corssValidationAU5).Count() / kfoldAU5)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 40 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU5(corssValidationAU5, countfoldAU5, kfoldAU5, kValue);

                        for (int i = kfoldAU5 * (countfoldAU5 - 1); i < kfoldAU5 * countfoldAU5; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU5)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Shock"))
                                {
                                    AU5YY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Sad"))
                                {
                                    AU5NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Laugh"))
                                {
                                    AU5YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Smile"))
                                {
                                    AU5NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Shock"))
                                {
                                    AU5YN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Sad"))
                                {
                                    AU5NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Laugh"))
                                {
                                    AU5YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Smile"))
                                {
                                    AU5NN++;
                                }
                            }
                        }
                        countfoldAU5++;
                    }

                    //for au6
                    while (countfoldAU6 <= balancedLibrary.balanceLibrary(corssValidationAU6).Count() / kfoldAU6)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 50 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU6(corssValidationAU6, countfoldAU6, kfoldAU6, kValue);

                        for (int i = kfoldAU6 * (countfoldAU6 - 1); i < kfoldAU6 * countfoldAU6; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU6)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Shock"))
                                {
                                    AU6NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Sad"))
                                {
                                    AU6NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Laugh"))
                                {
                                    AU6YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Smile"))
                                {
                                    AU6YY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Shock"))
                                {
                                    AU6NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Sad"))
                                {
                                    AU6NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Laugh"))
                                {
                                    AU6YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Smile"))
                                {
                                    AU6YN++;
                                }
                            }
                        }
                        countfoldAU6++;
                    }

                    //for au12

                    while (countfoldAU12 <= balancedLibrary.balanceLibrary(corssValidationAU12).Count() / kfoldAU12)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 60 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU12(corssValidationAU12, countfoldAU12, kfoldAU12, kValue);

                        for (int i = kfoldAU12 * (countfoldAU12 - 1); i < kfoldAU12 * countfoldAU12; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU12)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Shock"))
                                {
                                    AU12NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Sad"))
                                {
                                    AU12NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Laugh"))
                                {
                                    AU12NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Smile"))
                                {
                                    AU12YY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Shock"))
                                {
                                    AU12NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Sad"))
                                {
                                    AU12NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Laugh"))
                                {
                                    AU12NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Smile"))
                                {
                                    AU12YN++;
                                }
                            }
                        }
                        countfoldAU12++;
                    }

                    //for au15

                    while (countfoldAU15 <= balancedLibrary.balanceLibrary(corssValidationAU15).Count() / kfoldAU15)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 70 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU15(corssValidationAU15, countfoldAU15, kfoldAU15, kValue);

                        for (int i = kfoldAU15 * (countfoldAU15 - 1); i < kfoldAU15 * countfoldAU15; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU15)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Shock"))
                                {
                                    AU15NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Sad"))
                                {
                                    AU15YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Laugh"))
                                {
                                    AU15NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Smile"))
                                {
                                    AU15NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Shock"))
                                {
                                    AU15NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Sad"))
                                {
                                    AU15YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Laugh"))
                                {
                                    AU15NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Smile"))
                                {
                                    AU15NN++;
                                }
                            }
                        }
                        countfoldAU15++;
                    }

                    //for au17

                    while (countfoldAU17 <= balancedLibrary.balanceLibrary(corssValidationAU17).Count() / kfoldAU17)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 80 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU17(corssValidationAU17, countfoldAU17, kfoldAU17, kValue);

                        for (int i = kfoldAU17 * (countfoldAU17 - 1); i < kfoldAU17 * countfoldAU17; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU17)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Shock"))
                                {
                                    AU17NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Sad"))
                                {
                                    AU17YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Laugh"))
                                {
                                    AU17NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Smile"))
                                {
                                    AU17NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Shock"))
                                {
                                    AU17NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Sad"))
                                {
                                    AU17YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Laugh"))
                                {
                                    AU17NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Smile"))
                                {
                                    AU17NN++;
                                }
                            }
                        }
                        countfoldAU17++;
                    }

                    //for au26

                    while (countfoldAU26 <= balancedLibrary.balanceLibrary(corssValidationAU26).Count() / kfoldAU26)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 90 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU26(corssValidationAU26, countfoldAU26, kfoldAU26, kValue);

                        for (int i = kfoldAU26 * (countfoldAU26 - 1); i < kfoldAU26 * countfoldAU26; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU26)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Shock"))
                                {
                                    AU26NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Sad"))
                                {
                                    AU26NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Laugh"))
                                {
                                    AU26YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Smile"))
                                {
                                    AU26YY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Shock"))
                                {
                                    AU26NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Sad"))
                                {
                                    AU26NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Laugh"))
                                {
                                    AU26YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Smile"))
                                {
                                    AU26YN++;
                                }
                            }
                        }
                        countfoldAU26++;
                    }

                    //for au27

                    while (countfoldAU27 <= balancedLibrary.balanceLibrary(corssValidationAU27).Count() / kfoldAU27)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 95 + "%\n"));

                        KnnAUCossvalidation = knnModel.crossValidateAU27(corssValidationAU27, countfoldAU27, kfoldAU27, kValue);

                        for (int i = kfoldAU27 * (countfoldAU27 - 1); i < kfoldAU27 * countfoldAU27; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU27)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (KnnAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Shock"))
                                {
                                    AU27YY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Sad"))
                                {
                                    AU27NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Laugh"))
                                {
                                    AU27NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Smile"))
                                {
                                    AU27NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Shock"))
                                {
                                    AU27YN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Sad"))
                                {
                                    AU27NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Laugh"))
                                {
                                    AU27NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Smile"))
                                {
                                    AU27NN++;
                                }
                            }
                        }
                        countfoldAU27++;
                    }

                    Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 100 + "%\n"));
                    Dispatcher.Invoke((Action)(() => croseValidation.Text += "done building cross validation table using KNN,k is:" + kValue.ToString() + "\n"));
                    Dispatcher.Invoke((Action)(() => croseValidation.Text += "====================================\n"));

                    Dispatcher.Invoke((Action)(() =>
                     {
                         double accuracyAU1 = (Convert.ToDouble(AU1YY + AU1NN) / (AU1YY + AU1YN + AU1NY + AU1NN));
                         croseValidation.Text += "AU1" + "\n";
                         croseValidation.Text += AU1YY.ToString() + "      " + AU1YN.ToString() + "\n\n";
                         croseValidation.Text += AU1NY.ToString() + "      " + AU1NN.ToString() + "   " + "accuracy :" + accuracyAU1.ToString() + "\n";
                         croseValidation.Text += "====================================\n";


                         double accuracyAU2 = (Convert.ToDouble(AU2YY + AU2NN) / (AU2YY + AU2YN + AU2NY + AU2NN));
                         croseValidation.Text += "AU2" + "\n";
                         croseValidation.Text += AU2YY.ToString() + "      " + AU2YN.ToString() + "\n\n";
                         croseValidation.Text += AU2NY.ToString() + "      " + AU2NN.ToString() + "   " + "accuracy :" + accuracyAU2.ToString() + "\n";
                         croseValidation.Text += "====================================\n";


                         double accuracyAU4 = (Convert.ToDouble(AU4YY + AU4NN) / (AU4YY + AU4YN + AU4NY + AU4NN));
                         croseValidation.Text += "AU4" + "\n";
                         croseValidation.Text += AU4YY.ToString() + "      " + AU4YN.ToString() + "\n\n";
                         croseValidation.Text += AU4NY.ToString() + "      " + AU4NN.ToString() + "   " + "accuracy :" + accuracyAU4.ToString() + "\n";
                         croseValidation.Text += "====================================\n";

                         double accuracyAU5 = (Convert.ToDouble(AU5YY + AU5NN) / (AU5YY + AU5YN + AU5NY + AU5NN));
                         croseValidation.Text += "AU5" + "\n";
                         croseValidation.Text += AU5YY.ToString() + "      " + AU5YN.ToString() + "\n\n";
                         croseValidation.Text += AU5NY.ToString() + "      " + AU5NN.ToString() + "   " + "accuracy :" + accuracyAU5.ToString() + "\n";
                         croseValidation.Text += "====================================\n";

                         double accuracyAU6 = (Convert.ToDouble(AU6YY + AU6NN) / (AU6YY + AU6YN + AU6NY + AU6NN));
                         croseValidation.Text += "AU6" + "\n";
                         croseValidation.Text += AU6YY.ToString() + "      " + AU6YN.ToString() + "\n\n";
                         croseValidation.Text += AU6NY.ToString() + "      " + AU6NN.ToString() + "   " + "accuracy :" + accuracyAU6.ToString() + "\n";
                         croseValidation.Text += "====================================\n";

                         double accuracyAU12 = (Convert.ToDouble(AU12YY + AU12NN) / (AU12YY + AU12YN + AU12NY + AU12NN));
                         croseValidation.Text += "AU12" + "\n";
                         croseValidation.Text += AU12YY.ToString() + "      " + AU12YN.ToString() + "\n\n";
                         croseValidation.Text += AU12NY.ToString() + "      " + AU12NN.ToString() + "   " + "accuracy :" + accuracyAU12.ToString() + "\n";
                         croseValidation.Text += "====================================\n";

                         double accuracyAU15 = (Convert.ToDouble(AU15YY + AU15NN) / (AU15YY + AU15YN + AU15NY + AU15NN));
                         croseValidation.Text += "AU15" + "\n";
                         croseValidation.Text += AU15YY.ToString() + "      " + AU15YN.ToString() + "\n\n";
                         croseValidation.Text += AU15NY.ToString() + "      " + AU15NN.ToString() + "   " + "accuracy :" + accuracyAU15.ToString() + "\n";
                         croseValidation.Text += "====================================\n";

                         double accuracyAU17 = Convert.ToDouble((AU17YY + AU17NN) / (AU17YY + AU17YN + AU17NY + AU17NN));
                         croseValidation.Text += "AU17" + "\n";
                         croseValidation.Text += AU17YY.ToString() + "      " + AU17YN.ToString() + "\n\n";
                         croseValidation.Text += AU17NY.ToString() + "      " + AU17NN.ToString() + "   " + "accuracy :" + accuracyAU17.ToString() + "\n";
                         croseValidation.Text += "====================================\n";

                         double accuracyAU26 = Convert.ToDouble(AU26YY + AU26NN) / (AU26YY + AU26YN + AU26NY + AU26NN);
                         croseValidation.Text += "AU26" + "\n";
                         croseValidation.Text += AU26YY.ToString() + "      " + AU26YN.ToString() + "\n\n";
                         croseValidation.Text += AU26NY.ToString() + "      " + AU26NN.ToString() + "   " + "accuracy :" + accuracyAU26.ToString() + "\n";
                         croseValidation.Text += "====================================\n";

                         double accuracyAU27 = (Convert.ToDouble(AU27YY + AU27NN) / (AU27YY + AU27YN + AU27NY + AU27NN));
                         croseValidation.Text += "AU27" + "\n";
                         croseValidation.Text += AU27YY.ToString() + "      " + AU27YN.ToString() + "\n\n";
                         croseValidation.Text += AU27NY.ToString() + "      " + AU27NN.ToString() + "   " + "accuracy :" + accuracyAU27.ToString() + "\n";
                         croseValidation.Text += "====================================\n";

                         croseValidation.Text += "overall accuracy is :" + ((accuracyAU1 + accuracyAU2 + accuracyAU4 + accuracyAU5 + accuracyAU6 + accuracyAU12 + accuracyAU15 + accuracyAU17 + accuracyAU26 + accuracyAU27) / 10).ToString();
                         MessageBox.Show("ovaerall accuracy is :" + ((accuracyAU1 + accuracyAU2 + accuracyAU4 + accuracyAU5 + accuracyAU6 + accuracyAU12 + accuracyAU15 + accuracyAU17 + accuracyAU26 + accuracyAU27) / 10).ToString());
                         CrossValidateAU.IsEnabled = true;
                     }));
                });
            }
            else
            { //using svm
                ThreadPool.QueueUserWorkItem((o) =>
                {
                    Dispatcher.Invoke((Action)(() =>
                    {
                        CrossValidateAU.IsEnabled = false;
                    }));

                    while (countfold <= balancedLibrary.balanceLibrary(corssValidateAU1).Count() / kfold)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 10 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU1(corssValidateAU1, countfold, kfold, kValue);

                        for (int i = kfold * (countfold - 1); i < kfold * countfold; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidateAU1)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Shock"))
                                {
                                    AU1YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Sad"))
                                {
                                    AU1YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Laugh"))
                                {
                                    AU1NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Smile"))
                                {
                                    AU1NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Shock"))
                                {
                                    AU1YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Sad"))
                                {
                                    AU1YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Laugh"))
                                {
                                    AU1NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidateAU1)[i].Contains("Smile"))
                                {
                                    AU1NN++;
                                }
                            }
                        }
                        countfold++;

                    }

                    //for au2
                    while (countfoldAU2 <= balancedLibrary.balanceLibrary(corssValidationAU2).Count() / kfoldAU2)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 20 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU2(corssValidationAU2, countfoldAU2, kfoldAU2, kValue);

                        for (int i = kfoldAU2 * (countfoldAU2 - 1); i < kfoldAU2 * countfoldAU2; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU2)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Shock"))
                                {
                                    AU2YY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Sad"))
                                {
                                    AU2NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Laugh"))
                                {
                                    AU2YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Smile"))
                                {
                                    AU2NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Shock"))
                                {
                                    AU2YN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Sad"))
                                {
                                    AU2NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Laugh"))
                                {
                                    AU2YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU2)[i].Contains("Smile"))
                                {
                                    AU2NN++;
                                }
                            }
                        }
                        countfoldAU2++;
                    }

                    //for au4
                    while (countfoldAU4 <= balancedLibrary.balanceLibrary(corssValidationAU4).Count() / kfoldAU4)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 30 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU4(corssValidationAU4, countfoldAU4, kfoldAU4, kValue);

                        for (int i = kfoldAU4 * (countfoldAU4 - 1); i < kfoldAU4 * countfoldAU4; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU4)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Shock"))
                                {
                                    AU4NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Sad"))
                                {
                                    AU4YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Laugh"))
                                {
                                    AU4NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Smile"))
                                {
                                    AU4NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Shock"))
                                {
                                    AU4NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Sad"))
                                {
                                    AU4YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Laugh"))
                                {
                                    AU4NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU4)[i].Contains("Smile"))
                                {
                                    AU4NN++;
                                }
                            }
                        }
                        countfoldAU4++;
                    }

                    //for au5
                    while (countfoldAU5 <= balancedLibrary.balanceLibrary(corssValidationAU5).Count() / kfoldAU5)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 40 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU5(corssValidationAU5, countfoldAU5, kfoldAU5, kValue);

                        for (int i = kfoldAU5 * (countfoldAU5 - 1); i < kfoldAU5 * countfoldAU5; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU5)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Shock"))
                                {
                                    AU5YY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Sad"))
                                {
                                    AU5NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Laugh"))
                                {
                                    AU5YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Smile"))
                                {
                                    AU5NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Shock"))
                                {
                                    AU5YN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Sad"))
                                {
                                    AU5NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Laugh"))
                                {
                                    AU5YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU5)[i].Contains("Smile"))
                                {
                                    AU5NN++;
                                }
                            }
                        }
                        countfoldAU5++;
                    }

                    //for au6
                    while (countfoldAU6 <= balancedLibrary.balanceLibrary(corssValidationAU6).Count() / kfoldAU6)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 50 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU6(corssValidationAU6, countfoldAU6, kfoldAU6, kValue);

                        for (int i = kfoldAU6 * (countfoldAU6 - 1); i < kfoldAU6 * countfoldAU6; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU6)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Shock"))
                                {
                                    AU6NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Sad"))
                                {
                                    AU6NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Laugh"))
                                {
                                    AU6YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Smile"))
                                {
                                    AU6YY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Shock"))
                                {
                                    AU6NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Sad"))
                                {
                                    AU6NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Laugh"))
                                {
                                    AU6YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU6)[i].Contains("Smile"))
                                {
                                    AU6YN++;
                                }
                            }
                        }
                        countfoldAU6++;
                    }

                    //for au12

                    while (countfoldAU12 <= balancedLibrary.balanceLibrary(corssValidationAU12).Count() / kfoldAU12)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 60 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU12(corssValidationAU12, countfoldAU12, kfoldAU12, kValue);

                        for (int i = kfoldAU12 * (countfoldAU12 - 1); i < kfoldAU12 * countfoldAU12; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU12)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Shock"))
                                {
                                    AU12NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Sad"))
                                {
                                    AU12NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Laugh"))
                                {
                                    AU12NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Smile"))
                                {
                                    AU12YY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Shock"))
                                {
                                    AU12NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Sad"))
                                {
                                    AU12NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Laugh"))
                                {
                                    AU12NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU12)[i].Contains("Smile"))
                                {
                                    AU12YN++;
                                }
                            }
                        }
                        countfoldAU12++;
                    }

                    //for au15

                    while (countfoldAU15 <= balancedLibrary.balanceLibrary(corssValidationAU15).Count() / kfoldAU15)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 70 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU15(corssValidationAU15, countfoldAU15, kfoldAU15, kValue);

                        for (int i = kfoldAU15 * (countfoldAU15 - 1); i < kfoldAU15 * countfoldAU15; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU15)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Shock"))
                                {
                                    AU15NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Sad"))
                                {
                                    AU15YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Laugh"))
                                {
                                    AU15NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Smile"))
                                {
                                    AU15NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Shock"))
                                {
                                    AU15NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Sad"))
                                {
                                    AU15YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Laugh"))
                                {
                                    AU15NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU15)[i].Contains("Smile"))
                                {
                                    AU15NN++;
                                }
                            }
                        }
                        countfoldAU15++;
                    }

                    //for au17

                    while (countfoldAU17 <= balancedLibrary.balanceLibrary(corssValidationAU17).Count() / kfoldAU17)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 80 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU17(corssValidationAU17, countfoldAU17, kfoldAU17, kValue);

                        for (int i = kfoldAU17 * (countfoldAU17 - 1); i < kfoldAU17 * countfoldAU17; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU17)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Shock"))
                                {
                                    AU17NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Sad"))
                                {
                                    AU17YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Laugh"))
                                {
                                    AU17NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Smile"))
                                {
                                    AU17NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Shock"))
                                {
                                    AU17NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Sad"))
                                {
                                    AU17YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Laugh"))
                                {
                                    AU17NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU17)[i].Contains("Smile"))
                                {
                                    AU17NN++;
                                }
                            }
                        }
                        countfoldAU17++;
                    }

                    //for au26

                    while (countfoldAU26 <= balancedLibrary.balanceLibrary(corssValidationAU26).Count() / kfoldAU26)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 90 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU26(corssValidationAU26, countfoldAU26, kfoldAU26, kValue);

                        for (int i = kfoldAU26 * (countfoldAU26 - 1); i < kfoldAU26 * countfoldAU26; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU26)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Shock"))
                                {
                                    AU26NY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Sad"))
                                {
                                    AU26NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Laugh"))
                                {
                                    AU26YY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Smile"))
                                {
                                    AU26YY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Shock"))
                                {
                                    AU26NN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Sad"))
                                {
                                    AU26NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Laugh"))
                                {
                                    AU26YN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU26)[i].Contains("Smile"))
                                {
                                    AU26YN++;
                                }
                            }
                        }
                        countfoldAU26++;
                    }

                    //for au27

                    while (countfoldAU27 <= balancedLibrary.balanceLibrary(corssValidationAU27).Count() / kfoldAU27)
                    {
                        Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 95 + "%\n"));

                        SvmAUCossvalidation = svmModel.crossValidateAU27(corssValidationAU27, countfoldAU27, kfoldAU27, kValue);

                        for (int i = kfoldAU27 * (countfoldAU27 - 1); i < kfoldAU27 * countfoldAU27; i++)
                        {
                            XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                            TextReader reader = new StreamReader(balancedLibrary.balanceLibrary(corssValidationAU27)[i]);
                            var read = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);

                            if (SvmAUCossvalidation.Decide(read.Distances.ToArray()) == 1)
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Shock"))
                                {
                                    AU27YY++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Sad"))
                                {
                                    AU27NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Laugh"))
                                {
                                    AU27NY++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Smile"))
                                {
                                    AU27NY++;
                                }
                            }
                            else
                            {
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Shock"))
                                {
                                    AU27YN++;

                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Sad"))
                                {
                                    AU27NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Laugh"))
                                {
                                    AU27NN++;
                                }
                                if (balancedLibrary.balanceLibrary(corssValidationAU27)[i].Contains("Smile"))
                                {
                                    AU27NN++;
                                }
                            }
                        }
                        countfoldAU27++;
                    }

                    Dispatcher.Invoke((Action)(() => croseValidation.Text = "building cross validation table ....." + 100 + "%\n"));
                    Dispatcher.Invoke((Action)(() => croseValidation.Text += "done building cross validation table using SVM \n"));
                    Dispatcher.Invoke((Action)(() => croseValidation.Text += "====================================\n"));

                    Dispatcher.Invoke((Action)(() =>
                    {
                        double accuracyAU1 = (Convert.ToDouble(AU1YY + AU1NN) / (AU1YY + AU1YN + AU1NY + AU1NN));
                        croseValidation.Text += "AU1" + "\n";
                        croseValidation.Text += AU1YY.ToString() + "      " + AU1YN.ToString() + "\n\n";
                        croseValidation.Text += AU1NY.ToString() + "      " + AU1NN.ToString() + "   " + "accuracy :" + accuracyAU1.ToString() + "\n";
                        croseValidation.Text += "====================================\n";


                        double accuracyAU2 = (Convert.ToDouble(AU2YY + AU2NN) / (AU2YY + AU2YN + AU2NY + AU2NN));
                        croseValidation.Text += "AU2" + "\n";
                        croseValidation.Text += AU2YY.ToString() + "      " + AU2YN.ToString() + "\n\n";
                        croseValidation.Text += AU2NY.ToString() + "      " + AU2NN.ToString() + "   " + "accuracy :" + accuracyAU2.ToString() + "\n";
                        croseValidation.Text += "====================================\n";


                        double accuracyAU4 = (Convert.ToDouble(AU4YY + AU4NN) / (AU4YY + AU4YN + AU4NY + AU4NN));
                        croseValidation.Text += "AU4" + "\n";
                        croseValidation.Text += AU4YY.ToString() + "      " + AU4YN.ToString() + "\n\n";
                        croseValidation.Text += AU4NY.ToString() + "      " + AU4NN.ToString() + "   " + "accuracy :" + accuracyAU4.ToString() + "\n";
                        croseValidation.Text += "====================================\n";

                        double accuracyAU5 = (Convert.ToDouble(AU5YY + AU5NN) / (AU5YY + AU5YN + AU5NY + AU5NN));
                        croseValidation.Text += "AU5" + "\n";
                        croseValidation.Text += AU5YY.ToString() + "      " + AU5YN.ToString() + "\n\n";
                        croseValidation.Text += AU5NY.ToString() + "      " + AU5NN.ToString() + "   " + "accuracy :" + accuracyAU5.ToString() + "\n";
                        croseValidation.Text += "====================================\n";

                        double accuracyAU6 = (Convert.ToDouble(AU6YY + AU6NN) / (AU6YY + AU6YN + AU6NY + AU6NN));
                        croseValidation.Text += "AU6" + "\n";
                        croseValidation.Text += AU6YY.ToString() + "      " + AU6YN.ToString() + "\n\n";
                        croseValidation.Text += AU6NY.ToString() + "      " + AU6NN.ToString() + "   " + "accuracy :" + accuracyAU6.ToString() + "\n";
                        croseValidation.Text += "====================================\n";

                        double accuracyAU12 = (Convert.ToDouble(AU12YY + AU12NN) / (AU12YY + AU12YN + AU12NY + AU12NN));
                        croseValidation.Text += "AU12" + "\n";
                        croseValidation.Text += AU12YY.ToString() + "      " + AU12YN.ToString() + "\n\n";
                        croseValidation.Text += AU12NY.ToString() + "      " + AU12NN.ToString() + "   " + "accuracy :" + accuracyAU12.ToString() + "\n";
                        croseValidation.Text += "====================================\n";

                        double accuracyAU15 = (Convert.ToDouble(AU15YY + AU15NN) / (AU15YY + AU15YN + AU15NY + AU15NN));
                        croseValidation.Text += "AU15" + "\n";
                        croseValidation.Text += AU15YY.ToString() + "      " + AU15YN.ToString() + "\n\n";
                        croseValidation.Text += AU15NY.ToString() + "      " + AU15NN.ToString() + "   " + "accuracy :" + accuracyAU15.ToString() + "\n";
                        croseValidation.Text += "====================================\n";

                        double accuracyAU17 = Convert.ToDouble((AU17YY + AU17NN) / (AU17YY + AU17YN + AU17NY + AU17NN));
                        croseValidation.Text += "AU17" + "\n";
                        croseValidation.Text += AU17YY.ToString() + "      " + AU17YN.ToString() + "\n\n";
                        croseValidation.Text += AU17NY.ToString() + "      " + AU17NN.ToString() + "   " + "accuracy :" + accuracyAU17.ToString() + "\n";
                        croseValidation.Text += "====================================\n";

                        double accuracyAU26 = Convert.ToDouble(AU26YY + AU26NN) / (AU26YY + AU26YN + AU26NY + AU26NN);
                        croseValidation.Text += "AU26" + "\n";
                        croseValidation.Text += AU26YY.ToString() + "      " + AU26YN.ToString() + "\n\n";
                        croseValidation.Text += AU26NY.ToString() + "      " + AU26NN.ToString() + "   " + "accuracy :" + accuracyAU26.ToString() + "\n";
                        croseValidation.Text += "====================================\n";

                        double accuracyAU27 = (Convert.ToDouble(AU27YY + AU27NN) / (AU27YY + AU27YN + AU27NY + AU27NN));
                        croseValidation.Text += "AU27" + "\n";
                        croseValidation.Text += AU27YY.ToString() + "      " + AU27YN.ToString() + "\n\n";
                        croseValidation.Text += AU27NY.ToString() + "      " + AU27NN.ToString() + "   " + "accuracy :" + accuracyAU27.ToString() + "\n";
                        croseValidation.Text += "====================================\n";

                        croseValidation.Text += "overall accuracy is :" + ((accuracyAU1 + accuracyAU2 + accuracyAU4 + accuracyAU5 + accuracyAU6 + accuracyAU12 + accuracyAU15 + accuracyAU17 + accuracyAU26 + accuracyAU27) / 10).ToString();
                        MessageBox.Show("ovaerall accuracy is :" + ((accuracyAU1 + accuracyAU2 + accuracyAU4 + accuracyAU5 + accuracyAU6 + accuracyAU12 + accuracyAU15 + accuracyAU17 + accuracyAU26 + accuracyAU27) / 10).ToString());
                        CrossValidateAU.IsEnabled = true;
                    }));
                });
            }
        }



        ////////////////////////////
        /// functions for Tab page3
        ///////////////////////////

        /// <summary>
        /// load an expression for prediction
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Load_expression_Click(object sender, RoutedEventArgs e)
        {

            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();
            dlg.DefaultExt = ".kf"; // Default file extension
            dlg.Filter = "Kinect Face data |*.kf"; // Filter files by extension

            // Show open file dialog box
            Nullable<bool> result = dlg.ShowDialog();

            // Process open file dialog box results
            if (result == true)
            {
                bool isSucceed = true;
                XmlSerializer deserializer = new XmlSerializer(typeof(FaceFeature));
                TextReader reader = new StreamReader(dlg.FileName);
                getDsitanceToNoseTipInOneFrame one = new getDsitanceToNoseTipInOneFrame();

                try { one = ((FaceFeature)deserializer.Deserialize(reader)).averageDistance(); }
                catch (System.InvalidOperationException)
                {
                    isSucceed = false;
                    MessageBox.Show(dlg.FileName + "is not a valid file");
                }

                if (isSucceed)
                {
                    TestExpression.Text = "";
                    MessageBox.Show("Footage Sucessfully Loaded!");
                    TestExpression.Text = dlg.FileName;
                    guessButton.IsEnabled = true;
                }
                reader.Close();
            }


        }



        private void DataGridAU_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {

        }



        /// <summary>
        /// return the chosen action units in a list
        /// </summary>
        private List<int> aus()
        {
            List<int> allAUs = new List<int>();
            allAUs.Add(1); allAUs.Add(2); allAUs.Add(4); allAUs.Add(5); allAUs.Add(6); allAUs.Add(12);
            allAUs.Add(15); allAUs.Add(17); allAUs.Add(26); allAUs.Add(27);
            return allAUs;

        }




        /// <summary>
        /// make a guess on loaded expression 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GuessExpression_Click(object sender, RoutedEventArgs e)
        {
            results.Text = "";
            AUToExpressionTemplate templates = new AUToExpressionTemplate();
            double countSmile = 0;
            double countSad = 0;
            double countLaugh = 0;
            double countShock = 0;
            double smileCount = 0.0;
            double sadCount = 0;
            double laughCount = 0;
            double shockCount = 0;
            int kvalue = 0;
            XmlSerializer deserializer = new XmlSerializer(typeof(FaceFeature));
            try
            {
                TextReader reader = new StreamReader(TestExpression.Text);
                getDsitanceToNoseTipInOneFrame one = new getDsitanceToNoseTipInOneFrame();

                one = ((FaceFeature)deserializer.Deserialize(reader)).averageDistance();

                List<double> temp = new List<double>();
                if (KNNSelected.IsChecked == true)
                {

                    kvalue = Convert.ToInt32(((ComboBoxItem)ExpressionKvalue.SelectedItem).Content);
                    knnC1 = knnModel.KnnForAllAUs(kvalue)[0];
                    knnC2 = knnModel.KnnForAllAUs(kvalue)[1];
                    knnC4 = knnModel.KnnForAllAUs(kvalue)[2];
                    knnC5 = knnModel.KnnForAllAUs(kvalue)[3];
                    knnC6 = knnModel.KnnForAllAUs(kvalue)[4];
                    knnC12 = knnModel.KnnForAllAUs(kvalue)[5];
                    knnC15 = knnModel.KnnForAllAUs(kvalue)[6];
                    knnC17 = knnModel.KnnForAllAUs(kvalue)[7];
                    knnC26 = knnModel.KnnForAllAUs(kvalue)[8];
                    knnC27 = knnModel.KnnForAllAUs(kvalue)[9];
                    temp.Add(knnC1.Decide(one.Distances.ToArray()));
                    temp.Add(knnC2.Decide(one.Distances.ToArray()));
                    temp.Add(knnC4.Decide(one.Distances.ToArray()));
                    temp.Add(knnC5.Decide(one.Distances.ToArray()));
                    temp.Add(knnC6.Decide(one.Distances.ToArray()));
                    temp.Add(knnC12.Decide(one.Distances.ToArray()));
                    temp.Add(knnC15.Decide(one.Distances.ToArray()));
                    temp.Add(knnC17.Decide(one.Distances.ToArray()));
                    temp.Add(knnC26.Decide(one.Distances.ToArray()));
                    temp.Add(knnC27.Decide(one.Distances.ToArray()));
                    results.Text += "applying KNN classfier ,k is selected as :" + kvalue.ToString() + "\n";

                }
                else
                {
                    temp.Add(svm1.Decide(one.Distances.ToArray()));
                    temp.Add(svm2.Decide(one.Distances.ToArray()));
                    temp.Add(svm4.Decide(one.Distances.ToArray()));
                    temp.Add(svm5.Decide(one.Distances.ToArray()));
                    temp.Add(svm6.Decide(one.Distances.ToArray()));
                    temp.Add(svm12.Decide(one.Distances.ToArray()));
                    temp.Add(svm15.Decide(one.Distances.ToArray()));
                    temp.Add(svm17.Decide(one.Distances.ToArray()));
                    temp.Add(svm26.Decide(one.Distances.ToArray()));
                    temp.Add(svm27.Decide(one.Distances.ToArray()));
                    results.Text += "applying SVM linear classfier " + "\n";

                }

                Dictionary<string, List<double>> ActivatedAUss = new Dictionary<string, List<double>>();
                ActivatedAUss.Add("decision", temp);

                dataGridAU.ItemsSource = ActivatedAUss;

                for (int a = 0; a < templates.templateString().Count; a++)
                {

                    for (int k = 0; k < templates.templateString()[a].Count; k++)
                    {
                        for (int i = 0; i < ActivatedAUss["decision"].Count(); i++)
                        {

                            if (templates.templateString()[a][k] == aus()[i])
                            {
                                if (ActivatedAUss["decision"][i] == 1)
                                {
                                    if (a == 0)
                                    {
                                        countSmile++;
                                        smileCount += (1.0 / 6.0) * Convert.ToDouble(templates.templateString()[a].Count() - k);
                                    }
                                    if (a == 1)
                                    {
                                        countSad++;
                                        sadCount += (1.0 / 10.0) * Convert.ToDouble(templates.templateString()[a].Count() - k);
                                    }
                                    if (a == 2)
                                    {
                                        countShock++;
                                        shockCount += (1.0 / 10.0) * Convert.ToDouble(templates.templateString()[a].Count() - k);
                                    }
                                    if (a == 3)
                                    {
                                        countLaugh++;
                                        laughCount += (1.0 / 10.0) * Convert.ToDouble(templates.templateString()[a].Count() - k);
                                    }
                                }
                            }
                        }
                    }
                }
                results.Text += "==========================================" + "\n";
                results.Text += "sad:       " + countSad.ToString() + "         " + "weigh factors:" + "       " + sadCount.ToString() + "\n";
                results.Text += "shock:    " + countShock.ToString() + "        " + "weigh factors:" + "       " + shockCount.ToString() + "\n";
                results.Text += "smile:    " + countSmile.ToString() + "         " + "weigh factors:" + "       " + smileCount.ToString() + "\n";
                results.Text += "laugh:    " + countLaugh.ToString() + "         " + "weigh factors:" + "      " + laughCount.ToString() + "\n";
                var storeResult = new Dictionary<string, double>();
                storeResult.Add("sad", countSad * sadCount);
                storeResult.Add("shock", countShock * shockCount);
                storeResult.Add("smile", countSmile * smileCount);
                storeResult.Add("laugh", countLaugh * laughCount);

                var max = storeResult.Values.Max();
                var key = storeResult.FirstOrDefault(x => x.Value == max).Key;
                results.Text += "==========================================" + "\n";
                results.Text += "prediction result :" + key.ToString();

                if (key == "laugh")
                {
                    ExpressionImg.Source = ExpressionImgEnum[1];
                }
                if (key == "smile")
                {
                    ExpressionImg.Source = ExpressionImgEnum[0];

                }
                if (key == "shock")
                {
                    ExpressionImg.Source = ExpressionImgEnum[2];

                }
                if (key == "sad")
                {
                    ExpressionImg.Source = ExpressionImgEnum[3];

                }


            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }



        }



        private void GetLibrary_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        private void TextBox_Library_TextChanged(object sender, TextChangedEventArgs e)
        {

        }

        /// <summary>
        /// expression cross validation 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ExpressionValidation_Click(object sender, RoutedEventArgs e)
        {
            results.Text = "";

            double totalClassification = 0;
            List<string> wantedFiles = new List<string>();
            foreach (string file in Directory.EnumerateFiles(@"FFELibrary", "*.fp"))
            {
                wantedFiles.Add(file);
            }


            int[,] expressionCrossValidation = new int[4, 4];

            for (int i = 0; i < 4; i++)
            {
                for (int k = 0; k < 4; k++)
                {
                    expressionCrossValidation[i, k] = 0;
                }
            }
            Load_expression.IsEnabled = false;
            guessButton.IsEnabled = false;
            ExpressionValidation.IsEnabled = false;
            AUToExpressionTemplate templates = new AUToExpressionTemplate();
           
            ThreadPool.QueueUserWorkItem((o) =>
            {
             
                for (int c = 0; c < wantedFiles.Count(); c++)
                {
                    Dispatcher.Invoke((Action)(() => results.Text = "building cross validation table ....." + (100 * (c + 1) / wantedFiles.Count()).ToString() + "%\n"));
                    Dispatcher.Invoke((Action)(() => results.Text += "===========================================" + "\n"));

                    double countSmile = 0;
                    double countSad = 0;
                    double countLaugh = 0;
                    double countShock = 0;
                    double smileCount = 0.0;
                    double sadCount = 0;
                    double laughCount = 0;
                    double shockCount = 0;

                    XmlSerializer xxxxxserializer = new XmlSerializer(typeof(getDsitanceToNoseTipInOneFrame));
                    TextReader reader = new StreamReader(wantedFiles[c]);
                    var one = (getDsitanceToNoseTipInOneFrame)xxxxxserializer.Deserialize(reader);
                    List<int> temp = new List<int>();

                    Dispatcher.Invoke((Action)(() =>
                    {
                                          
                            temp.Add(svm1.Decide(one.Distances.ToArray()));
                            temp.Add(svm2.Decide(one.Distances.ToArray()));
                            temp.Add(svm4.Decide(one.Distances.ToArray()));
                            temp.Add(svm5.Decide(one.Distances.ToArray()));
                            temp.Add(svm6.Decide(one.Distances.ToArray()));
                            temp.Add(svm12.Decide(one.Distances.ToArray()));
                            temp.Add(svm15.Decide(one.Distances.ToArray()));
                            temp.Add(svm17.Decide(one.Distances.ToArray()));
                            temp.Add(svm26.Decide(one.Distances.ToArray()));
                            temp.Add(svm27.Decide(one.Distances.ToArray()));
                            results.Text += "expression cross validation using svm linear kernal:" + "\n";
                            results.Text += "===========================================" + "\n";
                        
                    }));

                    for (int a = 0; a < templates.templateString().Count; a++)
                    {
                        for (int k = 0; k < templates.templateString()[a].Count; k++)
                        {
                            for (int i = 0; i < temp.Count(); i++)
                            {
                                if (templates.templateString()[a][k] == aus()[i])
                                {
                                    if (temp[i] == 1)
                                    {
                                        if (a == 0)
                                        {
                                            countSmile++;
                                            smileCount += (1.0 / (k + 1) * k / 2) * Convert.ToDouble((templates.templateString()[a].Count() - k)); //calculate total weighting factors with the matched activated AU
                                        }
                                        if (a == 1)
                                        {
                                            countSad++;
                                            sadCount += (1.0 / (k + 1) * k / 2) * Convert.ToDouble((templates.templateString()[a].Count() - k));
                                        }
                                        if (a == 2)
                                        {
                                            countShock++;
                                            shockCount += (1.0 / (k + 1) * k / 2) * Convert.ToDouble((templates.templateString()[a].Count() - k));
                                        }
                                        if (a == 3)
                                        {
                                            countLaugh++;
                                            laughCount += (1.0 / (k + 1) * k / 2) * Convert.ToDouble((templates.templateString()[a].Count() - k));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    var storeResult = new Dictionary<string, double>();
                    storeResult.Add("Sad", countSad * sadCount);
                    storeResult.Add("Shock", countShock * shockCount);
                    storeResult.Add("Smile", countSmile * smileCount);
                    storeResult.Add("Laugh", countLaugh * laughCount);
                    var key = storeResult.FirstOrDefault(x => x.Value == storeResult.Values.Max()).Key; //find

                    if (wantedFiles[c].Contains(key))
                    {
                        if (key == "Smile")
                        {
                            expressionCrossValidation[0, 0]++;
                        }
                        if (key == "Laugh")
                        {
                            expressionCrossValidation[1, 1]++;
                        }
                        if (key == "Sad")
                        {
                            expressionCrossValidation[2, 2]++;
                        }
                        if (key == "Shock")
                        {
                            expressionCrossValidation[3, 3]++;
                        }
                    }

                    else
                    {
                        if (wantedFiles[c].Contains("Smile"))
                        {
                            if (key == "Laugh")
                            {
                                expressionCrossValidation[0, 1]++;

                            }
                            if (key == "Sad")
                            {
                                expressionCrossValidation[0, 2]++;

                            }
                            if (key == "Shock")
                            {
                                expressionCrossValidation[0, 3]++;

                            }
                        }
                        if (wantedFiles[c].Contains("Laugh"))
                        {
                            if (key == "Smile")
                            {
                                expressionCrossValidation[1, 0]++;
                            }
                            if (key == "Sad")
                            {
                                expressionCrossValidation[1, 2]++;

                            }
                            if (key == "Shock")
                            {
                                expressionCrossValidation[1, 3]++;

                            }

                        }
                        if (wantedFiles[c].Contains("Sad"))
                        {
                            if (key == "Smile")
                            {
                                expressionCrossValidation[2, 0]++;

                            }
                            if (key == "Laugh")
                            {
                                expressionCrossValidation[2, 1]++;

                            }
                            if (key == "Shock")
                            {
                                expressionCrossValidation[2, 3]++;

                            }
                        }
                        if (wantedFiles[c].Contains("Shock"))
                        {
                            if (key == "Smile")
                            {
                                expressionCrossValidation[3, 0]++;

                            }
                            if (key == "Laugh")
                            {
                                expressionCrossValidation[3, 1]++;

                            }
                            if (key == "Sad")
                            {
                                expressionCrossValidation[3, 2]++;

                            }
                        }

                    }

                }

                Dispatcher.Invoke((Action)(() =>
                {

                    results.Text += "              " + "Smile" + "    " + "Laugh" + "    " + "Sad" + "     " + "Shock" + "\n";
                    results.Text += "smile" + "       " + expressionCrossValidation[0, 0].ToString() + "            " + expressionCrossValidation[0, 1].ToString() + "            " + expressionCrossValidation[0, 2].ToString() + "            " + expressionCrossValidation[0, 3].ToString() + "\n";
                    results.Text += "laugh" + "       " + expressionCrossValidation[1, 0].ToString() + "           " + expressionCrossValidation[1, 1].ToString() + "             " + expressionCrossValidation[1, 2].ToString() + "            " + expressionCrossValidation[1, 3].ToString() + "\n";
                    results.Text += "sad  " + "        " + expressionCrossValidation[2, 0].ToString() + "            " + expressionCrossValidation[2, 1].ToString() + "            " + expressionCrossValidation[2, 2].ToString() + "             " + expressionCrossValidation[2, 3].ToString() + "\n";
                    results.Text += "shock" + "      " + expressionCrossValidation[3, 0].ToString() + "             " + expressionCrossValidation[3, 1].ToString() + "             " + expressionCrossValidation[3, 2].ToString() + "            " + expressionCrossValidation[3, 3].ToString() + "\n";
                    for (int i = 0; i < 4; i++)
                    {
                        for (int k = 0; k < 4; k++)
                        {
                            totalClassification += expressionCrossValidation[i, k];
                        }
                    }
                    results.Text += "Overall accuracy  :" + (Convert.ToDouble((expressionCrossValidation[0, 0] + expressionCrossValidation[1, 1] + expressionCrossValidation[2, 2] + expressionCrossValidation[3, 3])) / totalClassification).ToString() + "\n";
                    results.Text += "===========================================" + "\n";

                }
              ));
                Dispatcher.Invoke((Action)(() =>
                {
                    Load_expression.IsEnabled = true;
                    if (TestExpression.Text == "")
                    {
                        guessButton.IsEnabled = false;
                    }
                    else
                    {
                        guessButton.IsEnabled = true;
                    }

                    ExpressionValidation.IsEnabled = true;
                }
                    ));
            });
        }
        private void ExpressionValidation_TextChanged(object sender, TextChangedEventArgs e)
        {

        }



        ///////////////////////////
        /// self-defined function
        ///////////////////////////

        /// <summary>
        /// store the picture for each expression type
        /// </summary>  
        /// path may need to be changed on different computer
        static private BitmapImage[] ExpressionImgEnum =
       {
            new BitmapImage(new Uri(@"C:\Users\lijia\Desktop\FFEA Universe rights\Images\smile.png")),
            new BitmapImage(new Uri(@"C:\Users\lijia\Desktop\FFEA Universe rights\Images\laugh.png")),
            new BitmapImage(new Uri(@"C:\Users\lijia\Desktop\FFEA Universe rights\Images\shock.png")),
            new BitmapImage(new Uri(@"C:\Users\lijia\Desktop\FFEA Universe rights\Images\sad.png"))
        };

       
    }
}