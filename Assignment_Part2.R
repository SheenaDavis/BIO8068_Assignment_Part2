#Assignment Part 2 
# Automatic classification of taxa from photographs

#read in relevant packages 

necessary.packages<-c("devtools","behaviouR","tuneR","seewave","ggplot2","dplyr",
                      "warbleR","leaflet","lubridate","sp","sf","raster","mapview",
                      "leafem","BIRDS","xts","zoo", "stringr","vegan","rinat","keras")

already.installed <- necessary.packages%in%installed.packages()[,'Package'] #asks if the necessary packages are already installed
if (length(necessary.packages[!already.installed])>=1) { #if not installed download now
  install.packages(necessary.packages[!already.installed],dep=1)
}
sapply(necessary.packages, function(p){require(p,quietly = T,character.only = T)})

#to begin use download_images() function created in source script
source("download_images.R") 

#create bounding box using gb_simple.RDS
gb_ll <- readRDS("gb_simple (1).RDS")

#Beetles

#Alder Leaf  Beetle
alderleaf_recs <-  get_inat_obs(taxon_name  = "Agelastica alni",
                                bounds = gb_ll,
                                quality = "research",
                                # month=6,   # Month can be set.
                                # year=2018, # Year can be set.
                                maxresults = 600)

download_images(spp_recs = alderleaf_recs, spp_folder = "alderleaf")

#Thick-legged Flower Beetle
thick_leggedflower_recs <-  get_inat_obs(taxon_name  = "Oedemera nobilis",
                                 bounds = gb_ll,
                                 quality = "research",
                                 # month=6,   # Month can be set.
                                 # year=2018, # Year can be set.
                                 maxresults = 600)

download_images(spp_recs = thick_leggedflower_recs, spp_folder = "thick_leggedflower")

#Seven-spotted Ladybird
ladybird_recs <-  get_inat_obs(taxon_name  = "Coccinella septempunctata",
                                         bounds = gb_ll,
                                         quality = "research",
                                         # month=6,   # Month can be set.
                                         # year=2018, # Year can be set.
                                         maxresults = 600)

download_images(spp_recs = ladybird_recs, spp_folder = "ladybird")

##put test images in seperate folder

image_files_path <- "images" # path to folder with photos

# list of spp to model; these names must match folder names
spp_list <- dir(image_files_path) # Automatically pick up names
#spp_list <- c("ladybird", "thick_leggedflower", "alder_leaf") # manual entry

# number of spp classes (i.e. 3 species in this example)
output_n <- length(spp_list)

# Create test, and species sub-folders
for(folder in 1:output_n){
  dir.create(paste("test", spp_list[folder], sep="/"), recursive=TRUE)
}

# Now copy over spp_501.jpg to spp_600.jpg using two loops, deleting the photos
# from the original images folder after the copy
for(folder in 1:output_n){
  for(image in 501:600){
    src_image  <- paste0("images/", spp_list[folder], "/spp_", image, ".jpg")
    dest_image <- paste0("test/"  , spp_list[folder], "/spp_", image, ".jpg")
    file.copy(src_image, dest_image)
    file.remove(src_image)
  }
}

#Train up your deep learning model
# image size to scale down to (original images vary but about 400 x 500 px)
img_width <- 150
img_height <- 150
target_size <- c(img_width, img_height)

# Full-colour Red Green Blue = 3 channels
channels <- 3

#rescale your images (255 is max colour hue) and define the proportion (20%)
#that will be used for validation
# Rescale from 255 to between zero and 1

train_data_gen = image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)

#Reading all the images from a folder
# training images
train_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "training",
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "validation",
                                                    seed = 42)

cat("Number of images per class:")
table(factor(train_image_array_gen$classes))
cat("Class labels vs index mapping")
train_image_array_gen$class_indices

plot(as.raster(train_image_array_gen[[1]][[1]][8,,,]))

#Define additional parameters and configure model
# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32 # Useful to define explicitly as we'll use it later
epochs <- 10     # How long to keep training going for

#Now you define how your CNN is structured.
# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = "softmax") 

#Remember to check the CNN structure before compiling and running it
print(model)

#Define the error terms and accuracy measures. Use categorical_crossentropy as we have more than two species:
# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

#Set the deep-learning model off 
# Train the model using fit_generator
history <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
)

#plot the run history of the model
plot(history)

#Saving model for future use
# The imager package also has a save.image function, so unload it to
# avoid any confusion
detach("package:imager", unload = TRUE)

# The save.image function saves your whole R workspace
#save using appropriate file name 
save.image("beetles3.RData")

# Saves only the model, with all its weights and configuration, in a special
# hdf5 file on its own. You can use load_model_hdf5 to get it back.
#model %>% save_model_hdf5("animals_simple.hdf5")

#Testing your model
path_test <- "test"

test_data_gen <- image_data_generator(rescale = 1/255)

test_image_array_gen <- flow_images_from_directory(path_test,
                                                   test_data_gen,
                                                   target_size = target_size,
                                                   class_mode = "categorical",
                                                   classes = spp_list,
                                                   shuffle = FALSE, # do not shuffle the images around
                                                   batch_size = 1,  # Only 1 image at a time
                                                   seed = 123)

# Takes about 3 minutes to run through all the images
model %>% evaluate_generator(test_image_array_gen, 
                             steps = test_image_array_gen$n)



#predictions for our test photographs, but store the results in a data.frame
predictions <- model %>% 
  predict_generator(
    generator = test_image_array_gen,
    steps = test_image_array_gen$n
  ) %>% as.data.frame
colnames(predictions) <- spp_list

# Create 3 x 3 table to store data
confusion <- data.frame(matrix(0, nrow=3, ncol=3), row.names=spp_list)
colnames(confusion) <- spp_list

obs_values <- factor(c(rep(spp_list[1],100),
                       rep(spp_list[2], 100),
                       rep(spp_list[3], 100)))
pred_values <- factor(colnames(predictions)[apply(predictions, 1, which.max)])

#load caret library
library(caret)
#run confusion matrix for model sensitivity and specitivity.
conf_mat <- confusionMatrix(data = pred_values, reference = obs_values)
conf_mat

#Making a prediction for a single image
# Original image
test_image_plt <- imager::load.image("test/ladybird/spp_508.jpg")
plot(test_image_plt)

# Need to import slightly differently resizing etc. for Keras
test_image <- image_load("test/ladybird/spp_508.jpg",
                         target_size = target_size)

test_image <- image_to_array(test_image)
test_image <- array_reshape(test_image, c(1, dim(test_image)))
test_image <- test_image/255

# Now make the prediction, and print out nicely
pred <- model %>% predict(test_image)
pred <- data.frame("Species" = spp_list, "Probability" = t(pred))
pred <- pred[order(pred$Probability, decreasing=T),][1:3,]
pred$Probability <- paste(round(100*pred$Probability,2),"%")

#view predicted image 
pred
