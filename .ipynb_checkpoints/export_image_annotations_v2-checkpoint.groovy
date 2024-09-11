// https://qupath.readthedocs.io/en/latest/docs/advanced/exporting_images.html
// https://qupath.readthedocs.io/en/latest/docs/advanced/exporting_annotations.html#binary-labeled-images
// https://forum.image.sc/t/how-to-export-only-areas-within-annotations-as-whole-picture-or-tiles/39667/4

import qupath.lib.images.servers.LabeledImageServer

// Define output resolution in calibrated units (e.g. µm if available)
double requestedPixelSize = 0.465 //um to match olympus camera in pathologist's office at 20x 1920x1200px images

// Get the current image (supports 'Run for project')
def imageData = getCurrentImageData()

// Define output path (here, relative to project)
def imageName = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()).substring(0,12)

// Create output directory inside the project folder
def dirOutput = buildFilePath(PROJECT_BASE_DIR, 'image tiles')
mkdirs(dirOutput)

// Convert output resolution to a downsample factor
double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Tumor', 1)      // Choose output labels (the order matters!)
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labelled image servers
def exporter = new TileExporter(imageData)
    .downsample(downsample)   // Define export resolution
    .imageExtension('.jpg')   // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(640,640)            // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true) // If true, only export tiles if there is a (classified) annotation present
    .overlap(64)              // Define overlap, in pixel units at the export resolution
    .includePartialTiles(false)

// Uncomment this line if you want to see customization options
//println(describe(exporter))

// Write tiles to the specified directory
exporter.writeTiles(dirOutput)

print 'Done!'