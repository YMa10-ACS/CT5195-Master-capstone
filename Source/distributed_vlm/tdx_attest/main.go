package main

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/google/go-tdx-guest/abi"
	"github.com/google/go-tdx-guest/client"
	"github.com/google/go-tdx-guest/validate"
	"github.com/google/go-tdx-guest/verify"
)

const nonceSize = 64

type request struct {
	Nonce string `json:"nonce"`
	Quote string `json:"quote,omitempty"`
}

type response struct {
	Quote    string `json:"quote,omitempty"`
	Verified bool   `json:"verified,omitempty"`
}

func decodeBase64Field(name, value string) ([]byte, error) {
	if value == "" {
		return nil, fmt.Errorf("%s is required", name)
	}
	decoded, err := base64.StdEncoding.Strict().DecodeString(value)
	if err != nil {
		return nil, fmt.Errorf("%s is not valid Base64: %w", name, err)
	}
	return decoded, nil
}

func decodeNonce(value string) ([nonceSize]byte, error) {
	var nonce [nonceSize]byte
	decoded, err := decodeBase64Field("nonce", value)
	if err != nil {
		return nonce, err
	}
	if len(decoded) != nonceSize {
		return nonce, fmt.Errorf("nonce must be exactly %d bytes", nonceSize)
	}
	copy(nonce[:], decoded)
	return nonce, nil
}

func generateQuote(input request) (response, error) {
	nonce, err := decodeNonce(input.Nonce)
	if err != nil {
		return response{}, err
	}

	provider, err := client.GetQuoteProvider()
	if err != nil {
		return response{}, fmt.Errorf("get TDX quote provider: %w", err)
	}
	quote, err := client.GetRawQuote(provider, nonce)
	if err != nil {
		return response{}, fmt.Errorf("generate TDX quote: %w", err)
	}
	return response{Quote: base64.StdEncoding.EncodeToString(quote)}, nil
}

func verifyQuote(input request) (response, error) {
	nonce, err := decodeNonce(input.Nonce)
	if err != nil {
		return response{}, err
	}
	rawQuote, err := decodeBase64Field("quote", input.Quote)
	if err != nil {
		return response{}, err
	}
	quote, err := abi.QuoteToProto(rawQuote)
	if err != nil {
		return response{}, fmt.Errorf("parse TDX quote: %w", err)
	}

	if err := verify.TdxQuote(quote, verify.DefaultOptions()); err != nil {
		return response{}, fmt.Errorf("verify TDX quote signature: %w", err)
	}
	validationOptions := &validate.Options{
		TdQuoteBodyOptions: validate.TdQuoteBodyOptions{
			ReportData: nonce[:],
		},
	}
	if err := validate.TdxQuote(quote, validationOptions); err != nil {
		return response{}, fmt.Errorf("validate TDX quote nonce: %w", err)
	}
	return response{Verified: true}, nil
}

func run() error {
	if len(os.Args) != 2 {
		return errors.New("usage: tdx-attest <quote|verify>")
	}

	var input request
	decoder := json.NewDecoder(os.Stdin)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		return fmt.Errorf("decode request JSON: %w", err)
	}

	var (
		output response
		err    error
	)
	switch os.Args[1] {
	case "quote":
		output, err = generateQuote(input)
	case "verify":
		output, err = verifyQuote(input)
	default:
		return fmt.Errorf("unsupported command %q", os.Args[1])
	}
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(output)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
