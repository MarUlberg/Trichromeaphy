# Trichromeaphy

A simple Python utility for reconstructing color images from monochrome RGB exposures, without the need for Photoshop or other complicated software. Mainly created for GameBoy Camera pictures, but should work with other monochrome cameras aswell.

## Features

* Drag-and-drop support (Windows)
* Detects channels from `.r`, `.g`, `.b`, and optional `.l` subextensions
* Automatic image alignment

## Usage

Capture a monochrome image through red, green, and blue filters:

```
flower.r.jpg
flower.g.jpg
flower.b.jpg
```

Optionally include an unfiltered luminance image:

```
flower.l.jpg
```

Then drag all images onto `trichromeaphy.py`:


The output will be:

* `flower.rgb.jpg`

## License

MIT
