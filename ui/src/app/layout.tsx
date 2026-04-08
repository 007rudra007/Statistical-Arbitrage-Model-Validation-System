import './globals.css'
import React from 'react'

export const metadata = {
  title: 'Quant Terminal',
  description: 'Pairs Trading System UI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}
