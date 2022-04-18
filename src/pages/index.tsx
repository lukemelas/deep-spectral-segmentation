import { Link as ChakraLink, Text, Code, ListItem, Heading, UnorderedList } from '@chakra-ui/react'
import { Title, Authors } from 'components/Header'
import { Container } from 'components/Container'
import NextLink from 'next/link'
import { DarkModeSwitch } from 'components/DarkModeSwitch'
import { LinksRow } from 'components/LinksRow'
import { Footer } from 'components/Footer'

import { title, abstract, citationId, citationAuthors, citationYear, citationBooktitle, acknowledgements, video_url } from 'data'


const Index = () => (
  <Container>

    {/* Heading */}
    <Title />
    <Authors />

    {/* Links */}
    <LinksRow />

    {/* Video */}
    {/* <Container w="90vw" h="50.6vw" maxW="700px" maxH="393px" mb="3rem">
      <iframe
        width="100%" height="100%"
        src={video_url}
        title="Video"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen>
      </iframe>
    </Container> */}

    {/* Main */}
    <Container w="100%" maxW="44rem" alignItems="left" pl="1rem" pr="1rem">

      {/* Abstract */}
      <Heading fontSize="2xl" pb="1rem">Abstract</Heading>
      <Text pb="2rem">{abstract}</Text>

      {/* Example */}
      <Heading fontSize="2xl" pb="1rem">Examples</Heading>
      <img src={`${process.env.BASE_PATH || ""}/images/example.png`} />
      <Text align="center" pt="0.5rem" pb="0.5rem" fontSize="small">
        We present a simple approach based on spectral methods that decomposes an image using the eigenvectors of a Laplacian matrix constructed from a combination of color information and unsupervised deep features. Above, we show examples of these eigenvectors along with the results of our method on unsupervised object localization and semantic segmentation. 
      </Text>

      {/* Another Section */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem" id="dataset">Method</Heading>
      <img src={`${process.env.BASE_PATH || ""}/images/method.png`} />
      <Text align="center" pt="0.5rem" pb="0.5rem" fontSize="small">
        Our method first utilizes a self-supervised network to extract dense features corresponding to image patches. We then construct a weighted graph over patches, where edge weights give the semantic affinity of pairs of patches, and we consider the eigendecomposition of this graph's Laplacian matrix. We find that without imposing any additional structure, the eigenvectors of the Laplacian of this graph directly correspond to semantically meaningful image regions. 
      </Text>
      {/* <Text >Here we have...</Text> */}

      {/* Citation */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Citation</Heading>
      <Code p="0.5rem" borderRadius="5px" overflow="scroll" whiteSpace="nowrap">  {/*  fontFamily="monospace" */}
        @inproceedings&#123; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;{citationId}, <br />
          &nbsp;&nbsp;&nbsp;&nbsp;title=&#123;{title}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;author=&#123;{citationAuthors}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;year=&#123;{citationYear}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;booktitle=&#123;{citationBooktitle}&#125; <br />
        &#125;
      </Code>

      {/* Acknowledgements */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Acknowledgements</Heading>
      <Text >
        {acknowledgements}
      </Text>
    </Container>

    <DarkModeSwitch />
    <Footer>
      <Text></Text>
    </Footer>
  </Container >
)

export default Index