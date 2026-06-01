package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"
)

const (
	imagesPath = "data/t10k-images-idx3-ubyte"
	labelsPath = "data/t10k-labels-idx1-ubyte"
)

func loadFashionMnistImages(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 16)
	if _, err := f.Read(header); err != nil {
		return nil, err
	}

	numImages := int(binary.BigEndian.Uint32(header[4:8]))
	rows := int(binary.BigEndian.Uint32(header[8:12]))
	cols := int(binary.BigEndian.Uint32(header[12:16]))

	pixelsPerImage := rows * cols
	images := make([][]float64, numImages)
	for i := 0; i < numImages; i++ {
		imgData := make([]byte, pixelsPerImage)
		if _, err := f.Read(imgData); err != nil {
			return nil, err
		}
		images[i] = make([]float64, pixelsPerImage)
		for j := 0; j < pixelsPerImage; j++ {
			val := float64(imgData[j]) / 255.0
			images[i][j] = (val - 0.1307) / 0.3081
		}
	}
	return images, nil
}

func loadFashionMnistLabels(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 8)
	if _, err := f.Read(header); err != nil {
		return nil, err
	}

	numLabels := int(binary.BigEndian.Uint32(header[4:8]))
	labels := make([]int, numLabels)
	labelData := make([]byte, numLabels)
	if _, err := f.Read(labelData); err != nil {
		return nil, err
	}
	for i := 0; i < numLabels; i++ {
		labels[i] = int(labelData[i])
	}
	return labels, nil
}

func main() {
	fmt.Println("Loading data...")
	images, err := loadFashionMnistImages(imagesPath)
	if err != nil {
		panic(err)
	}

	labels, err := loadFashionMnistLabels(labelsPath)
	if err != nil {
		panic(err)
	}
	fmt.Println("Data loaded.")

	fmt.Println("Starting setup...")
	startSetup := time.Now()
	btEv, ev, params, encoder, encryptor, decryptor := fashion_mnist__configure()
	fmt.Printf("Setup complete. Took %v\n", time.Since(startSetup))

	total := 3
	correct := 0

	for i := 0; i < total; i++ {
		input := images[i]
		label := labels[i]

		inputFloat32 := make([]float32, len(input))
		for j := 0; j < len(input); j++ {
			inputFloat32[j] = float32(input[j])
		}

		fmt.Printf("Processing sample %d...\n", i)

		fmt.Println("Starting encryption...")
		startEncrypt := time.Now()
		ctInput := fashion_mnist__encrypt__arg0(ev, params, encoder, encryptor, inputFloat32)
		fmt.Printf("Encryption complete. Took %v\n", time.Since(startEncrypt))

		fmt.Println("Starting inference...")
		startTime := time.Now()
		resCt := fashion_mnist(btEv, ev, params, encoder, ctInput)
		duration := time.Since(startTime)
		fmt.Printf("Inference complete. Took %v\n", duration)

		fmt.Println("Starting decryption...")
		startDecrypt := time.Now()
		resValues := fashion_mnist__decrypt__result0(ev, params, encoder, decryptor, resCt)
		fmt.Printf("Decryption complete. Took %v\n", time.Since(startDecrypt))

		maxVal := float32(-math.MaxFloat32)
		maxIdx := -1
		for j := 0; j < 10; j++ {
			if resValues[j] > maxVal {
				maxVal = resValues[j]
				maxIdx = j
			}
		}

		if maxIdx == label {
			correct++
		}
		fmt.Printf("Sample %d: predicted %d, actual %d\n", i, maxIdx, label)
		fmt.Println("logits:")
		for j := 0; j < 10; j++ {
			fmt.Printf("%d, %.6f\n", j, resValues[j])
		}
	}

	accuracy := float64(correct) / float64(total)
	fmt.Printf("Accuracy: %.2f (%d/%d correct)\n", accuracy, correct, total)
}
