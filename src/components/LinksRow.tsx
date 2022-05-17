import { Button, Stack } from '@chakra-ui/react'
import { AiOutlineGithub, AiOutlineCodeSandbox, AiOutlineContainer } from "react-icons/ai"
import { IoIosPaper } from "react-icons/io"
import NextLink from 'next/link'

import { links } from 'data'

export const LinksRow = () => (
  <Stack direction="row" spacing={4} pt="2rem" pb="2rem">
    <NextLink href={links.github} passHref={true}>
      <Button leftIcon={<IoIosPaper size="25px" />} colorScheme="teal" variant="outline">
        Paper
      </Button>
    </NextLink>
    <NextLink href={links.github} passHref={true}>
      <Button leftIcon={<AiOutlineGithub size="25px" />} colorScheme="teal" variant="solid">
        Code
      </Button>
    </NextLink>
    <NextLink href={links.paper} passHref={true}>
      <Button leftIcon={<AiOutlineCodeSandbox size="25px" />} colorScheme="teal" variant="outline">
        Demo
      </Button>
    </NextLink>
    <NextLink href={links.poster} passHref={true}>
      <Button leftIcon={<AiOutlineContainer size="25px" />} colorScheme="teal" variant="solid">
        Poster
      </Button>
    </NextLink>
  </Stack >
)

