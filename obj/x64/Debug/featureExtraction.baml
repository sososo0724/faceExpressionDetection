   M S B A M L   `   `   `  ����H  DFFEA_Universe, Version=0.0.0.0, Culture=neutral, PublicKeyToken=null]3clr-namespace:Microsoft.Samples.Kinect.HDFaceBasics%Microsoft.Samples.Kinect.HDFaceBasics  5      =    7Microsoft.Samples.Kinect.HDFaceBasics.featureExtraction   5   	   W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5      R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5      Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5      @d2http://schemas.microsoft.com/expression/blend/2008     5      Elocal3clr-namespace:Microsoft.Samples.Kinect.HDFaceBasics     5      $	��700��5      $��1003.333��6,   .��5      � $��0,0,3.667,0.667q�6   .+�5	   
   &   System.Windows.Controls.DataGrid  -   5           Name$  dataGrid��5	      $
��Left=�6%   $	��505��6@   $��	10,40,0,0q�6M   $	��Top=�6`   $	��680��6x      AutoGenerateColumns .  ��6�   �� *getOneFootage+   Columns 5
      0  *System.Windows.Controls.DataGridTextColumn  5      ,  &System.Windows.Controls.DataGridColumn   Header$ Frames��6%   1  +System.Windows.Controls.DataGridBoundColumn   Binding 65   �� *Key+5      �� 5   
   -   5         ��Name$ totalDistance��5      $��calculate total��6(   $
��Left=�6`   $��795,605,0,0q�6{   $	��Top=�6�   $	��105��6�   $��35��6�   �� 5   
   -   5         ��Name$ groupBox��5      $��total distance moved��6%   $
��Left=�6C   $	��535��6^   $��
720,10,0,0q�6k   $	��Top=�6   $	��245��6�   .��5        -   5        $  	dataGrid1��5      $
��Left=�6*   $	��500��6E   $��0,10,0,0q�6R   $	��Top=�6d   $	��220��6|   ��6�   �� *
getTotal+5      5      �� 5   
   -   5         ��Name$ docName��5      $
��Left=�6"   $��30,8,0,0q�6=   $	��Top=�6O   $	��645��6g   $��27��6s   5      5      