import React, { useState } from 'react';
import {
  ChakraProvider,
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  Select,
  Text,
  Spinner,
  Heading,
  Flex,
  Image,
} from '@chakra-ui/react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState([]);
  const [logoImageUrl, setLogoImageUrl] = useState('');
  const [controlnetType, setControlnetType] = useState('canny_edge');
  const [numSteps, setNumSteps] = useState(20);
  const [loading, setLoading] = useState(false);
  const [generatedImage, setGeneratedImage] = useState('');

  const handleGenerate = async () => {
    setLoading(true);
    setGeneratedImage('');

    try {
      const response = await fetch('http://127.0.0.1:8000/generate_images', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          negative_prompt: negativePrompt,
          logo_image_url: logoImageUrl,
          controlnet_type: controlnetType,
          num_steps: numSteps,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate image. Status: ${response.status}`);
      }

      // Read the response as ArrayBuffer
      const buffer = await response.arrayBuffer();

      // Convert ArrayBuffer to Base64
      const base64Image = btoa(
        new Uint8Array(buffer).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ''
        )
      );

      // Set the generated image
      setGeneratedImage(`data:image/png;base64, ${base64Image}`);
    } catch (error) {
      console.error(error);

      // ... (handle errors)
    } finally {
      setLoading(false);
    }
  };

  return (
    <ChakraProvider>
      <Box
        minHeight="100vh"
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        p={8}
        backgroundColor="white"
        borderRadius="md"
        boxShadow="0px 0px 10px rgba(0, 0, 0, 0.1)"
      >
        <Heading mb={6} textAlign="center">
          Image Generation
        </Heading>
        <Flex
          width="100%"
          justifyContent="center"
          alignItems="center"
          flexWrap="wrap"
        >
          <Box width={['100%', '48%']}
            mb={4}
            mr={[0, '4%']}
            textAlign="center"
            backgroundColor="white"
            p={6}
            borderRadius="md"
            boxShadow="md">
            <FormControl>
              <FormLabel>Prompt</FormLabel>
              <Input
                placeholder="Enter prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </FormControl>

            <FormControl mt={4}>
              <FormLabel>Negative Prompt</FormLabel>
              <Input
                placeholder="Enter negative prompt, separate using commas"
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
              />
            </FormControl>


            <FormControl mt={4}>
              <FormLabel>Image URL</FormLabel>
              <Input
                placeholder="Enter logo image URL"
                value={logoImageUrl}
                onChange={(e) => setLogoImageUrl(e.target.value)}
              />
            </FormControl>

            <FormControl mt={4}>
              <FormLabel>Controlnet Type</FormLabel>
              <Select
                value={controlnetType}
                onChange={(e) => setControlnetType(e.target.value)}
              >
                <option value="canny_edge">Canny Edge</option>
                <option value="pose">Open Pose</option>
              </Select>
            </FormControl>

            <FormControl mt={4}>
              <FormLabel>Number of Steps</FormLabel>
              <Input
                type="number"
                value={numSteps}
                onChange={(e) => setNumSteps(e.target.value)}
              />
            </FormControl>

            <Button
              colorScheme="teal"
              onClick={handleGenerate}
              isLoading={loading}
              mt={4}
            >
              Generate
            </Button>
          </Box>

          {loading && !generatedImage && (
            <Box width={['100%', '48%']} mt={[4, 0]} display="flex" alignItems="center" justifyContent="center">
              <Spinner size="xl" color="teal.500" />
              <Text ml={4} color="gray.600">
                Generating Image...
              </Text>
            </Box>
          )}

          {generatedImage && (
            <Box width="48%" >
              <Image
                src={generatedImage}
                alt=""
                borderRadius="md"
                boxShadow="md"
              />
            </Box>
          )}
        </Flex>
      </Box>
    </ChakraProvider>
  );
}

export default App;