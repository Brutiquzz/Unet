def __init__(self):
  super(Unet, self).__init__()

  # Down hill1
  self.conv1 = nn.Conv3d(1, 2, kernel_size=3,  stride=1)
  self.conv2 = nn.Conv3d(2, 2, kernel_size=3,  stride=1)

  # Down hill2
  self.conv3 = nn.Conv3d(2, 4, kernel_size=3,  stride=1)
  self.conv4 = nn.Conv3d(4, 4, kernel_size=3,  stride=1)

  #bottom
  self.convbottom1 = nn.Conv3d(4, 8, kernel_size=3,  stride=1)
  self.convbottom2 = nn.Conv3d(8, 8, kernel_size=3,  stride=1)

  # Half upsampling
  self.halfconv1 = nn.Conv3d(8, 4, kernel_size=1, stride=1)

  #up hill4
  self.upConv0 = nn.Conv3d(8, 4, kernel_size=3,  stride=1)
  self.upConv1 = nn.Conv3d(4, 4, kernel_size=3,  stride=1)

  # half upsampling
  self.halfconv2 = nn.Conv3d(4, 2, kernel_size=1, stride=1)

  #up hill5
  self.upConv2 = nn.Conv3d(4, 2, kernel_size=3,  stride=1)
  self.upConv3 = nn.Conv3d(2, 2, kernel_size=3, stride=1)

  # down to 1 weight
  self.upConv4 = nn.Conv3d(2, 1, kernel_size=1, stride=1)

  self.mp = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

  #properties to be used:
  self.eval = []
  self.nextEpoch = True

  # state save
  self.bestState = 0

  # layers
  self.output1 = 0
  self.output2 = 0
  self.output3 = 0
  self.output4 = 0
  self.output5 = 0
  self.output6 = 0
  self.output7 = 0
  self.output8 = 0
  self.output9 = 0
  self.output10 = 0
  self.output11 = 0
  self.output12 = 0
  self.output13 = 0
  self.output14 = 0
  self.output15 = 0
  self.output16 = 0
  self.output17 = 0
  self.output18 = 0



# Use this function to update the filters of the object
def forward(self, input):
   # Use U-net Theory to Update the filters.
   # Example Approach...
   self.output1 = F.relu(self.conv1(input))
   self.output2 = F.relu(self.conv2(self.output1))

   self.output3 = self.mp(self.output2)

   self.output4 = F.relu(self.conv3(self.output3))
   self.output5 = F.relu(self.conv4(self.output4))

   self.output6 = self.mp(self.output5)

   self.output7 = F.relu(self.convbottom1(self.output6))
   self.output8 = F.relu(self.convbottom2(self.output7))

   self.output9 = F.interpolate(self.output8, scale_factor=2, mode='trilinear')
   self.output10 = F.relu(self.halfconv1(self.output9))
   self.output11 = torch.cat((self.output10, self.output5[:,:, 5:23, 5:23, 5:23]), 1)

   self.output12 = F.relu(self.upConv0(self.output11))
   self.output13 = F.relu(self.upConv1(self.output12))

   self.output14 = F.interpolate(self.output13, scale_factor=2, mode='trilinear')
   self.output15 = F.relu(self.halfconv2(self.output14))
   self.output16 = torch.cat((self.output15, self.output2[:,:, 17:45, 17:45, 17:45]), 1)

   self.output17 = F.relu(self.upConv2(self.output16))
   self.output18 = F.relu(self.upConv3(self.output17))

   return F.relu(self.upConv4(self.output18))


   def __init__(self):
     super(Unet, self).__init__()

     # Down hill1
     self.conv1 = nn.Conv3d(1, 2, kernel_size=3,  stride=1)
     self.conv2 = nn.Conv3d(2, 2, kernel_size=3,  stride=1)

     # Down hill2
     self.conv3 = nn.Conv3d(2, 4, kernel_size=3,  stride=1)
     self.conv4 = nn.Conv3d(4, 4, kernel_size=3,  stride=1)

     #bottom
     self.convbottom1 = nn.Conv3d(4, 8, kernel_size=3,  stride=1)
     self.convbottom2 = nn.Conv3d(8, 8, kernel_size=3,  stride=1)

     # Half upsampling
     self.halfconv1 = nn.Conv3d(8, 4, kernel_size=2, stride=1)

     #up hill4
     self.upConv0 = nn.Conv3d(8, 4, kernel_size=3,  stride=1)
     self.upConv1 = nn.Conv3d(4, 4, kernel_size=3,  stride=1)

     # half upsampling
     self.halfconv1 = nn.Conv3d(4, 2, kernel_size=2, stride=1)

     #up hill5
     self.upConv2 = nn.Conv3d(4, 2, kernel_size=3,  stride=1)
     self.upConv3 = nn.Conv3d(2, 2, kernel_size=3, stride=1)

     # down to 1 weight
     self.upConv4 = nn.Conv3d(2, 1, kernel_size=1, stride=1)

     self.mp = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

     #properties to be used:
     self.eval = []
     self.nextEpoch = True

     # state save
     self.bestState = 0

     # layers
     self.output1 = 0
     self.output2 = 0
     self.output3 = 0
     self.output4 = 0
     self.output5 = 0
     self.output6 = 0
     self.output7 = 0
     self.output8 = 0
     self.output9 = 0
     self.output10 = 0
     self.output11 = 0
     self.output12 = 0
     self.output13 = 0
     self.output14 = 0



# Use this function to update the filters of the object
   def forward(self, input):
      # Use U-net Theory to Update the filters.
      # Example Approach...
      self.output1 = F.relu(self.conv1(input))
      self.output2 = F.relu(self.conv2(self.output1))

      self.output3 = self.mp(self.output2)

      self.output4 = F.relu(self.conv3(self.output3))
      self.output5 = F.relu(self.conv4(self.output4))

      self.output6 = self.mp(self.output5)

      self.output7 = F.relu(self.convbottom1(self.output6))
      self.output8 = F.relu(self.convbottom2(self.output7))

      self.output9 = F.interpolate(self.output8, scale_factor=2, mode='trilinear')

      self.output10 = F.relu(self.upConv0(self.output9))
      self.output11 = F.relu(self.upConv1(self.output10))

      self.output12 = F.interpolate(self.output11, scale_factor=2, mode='trilinear')


      self.output13 = F.relu(self.upConv2(self.output12))
      self.output14 = F.relu(self.upConv3(self.output13))

      return F.relu(self.upConv4(self.output14))
