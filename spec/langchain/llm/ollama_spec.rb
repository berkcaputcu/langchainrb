# frozen_string_literal: true

require "faraday"

RSpec.describe Langchain::LLM::Ollama do
  let(:default_url) { "http://localhost:11434" }
  let(:subject) { described_class.new(url: default_url, default_options: {completion_model: "llama3.2", embedding_model: "llama3.2"}) }
  let(:client) { subject.send(:client) }

  describe "#initialize" do
    it "initializes the client without any errors" do
      expect { subject }.not_to raise_error
    end

    it "initialize with default arguments" do
      expect { described_class.new }.not_to raise_error
      expect(described_class.new.url).to eq(default_url)
    end

    it "sets auth headers if api_key is passed" do
      subject = described_class.new(url: "http://localhost)", api_key: "abc123")

      expect(subject.send(:client).headers).to include("Authorization" => "Bearer abc123")
    end

    context "when default_options are passed" do
      let(:default_options) { {response_format: "json", options: {num_ctx: 8_192}} }
      let(:messages) { [{role: "user", content: "Return data from the following sentence: John is a 30 year old software engineer living in SF."}] }
      let(:response) { subject.chat(messages: messages) { |resp| streamed_responses << resp } }
      let(:streamed_responses) { [] }

      subject { described_class.new(default_options: default_options) }

      it "sets the defaults options" do
        expect(subject.defaults[:response_format]).to eq("json")
        expect(subject.defaults[:options]).to eq(num_ctx: 8_192)
      end

      it "get passed to consecutive chat() call", vcr: {record: :once} do
        expect(client).to receive(:post).with("api/chat", hash_including(format: "json", options: {num_ctx: 8_192})).and_call_original
        expect(JSON.parse(response.chat_completion)).to eq({"name" => "John", "age" => 30, "occupation" => "software engineer", "location" => "SF"})
      end
    end
  end

  describe "#embed" do
    let(:response_body) {
      {"embeddings" => [[0.1, 0.2, 0.3]]}
    }

    before do
      allow_any_instance_of(Faraday::Connection).to receive(:post).and_return(double(body: response_body))
    end

    it "returns an embedding" do
      expect(subject.embed(text: "Hello, world!")).to be_a(Langchain::LLM::OllamaResponse)
      expect(subject.embed(text: "Hello, world!").embedding.count).to eq(3)
    end

    it "sends an input array to the client" do
      subject.embed(text: "Hello, world!")

      expect(client).to have_received(:post) do |path, &block|
        req = double("request").as_null_object
        block.call(req)
        expect(req).to have_received(:body=).with(hash_including(input: ["Hello, world!"]))
      end
    end

    context "when the JSON response contains no embeddings" do
      let(:response_body) {
        {"embeddings" => []}
      }

      it "#embedding returns nil" do
        expect(subject.embed(text: "Hello, world!").embedding).to be nil
      end
    end
  end

  describe "#complete" do
    let(:prompt) { "In one word, life is " }
    let(:response) { subject.complete(prompt: prompt) }

    it "returns a completion", :vcr do
      expect(response).to be_a(Langchain::LLM::OllamaResponse)
      expect(response.completion).to eq("Complicated.")
    end

    it "does not use streamed responses", vcr: {cassette_name: "Langchain_LLM_Ollama_complete_returns_a_completion"} do
      expect(client).to receive(:post).with("api/generate", hash_including(stream: false)).and_call_original
      response
    end

    context "when passing a block" do
      let(:response) { subject.complete(prompt: prompt) { |resp| streamed_responses << resp } }
      let(:streamed_responses) { [] }

      it "returns a completion", :vcr do
        expect(response).to be_a(Langchain::LLM::OllamaResponse)
        expect(response.completion).to eq("Complicated.")
        expect(response.total_tokens).to eq(36)
      end

      it "uses streamed responses", vcr: {cassette_name: "Langchain_LLM_Ollama_complete_when_passing_a_block_returns_a_completion"} do
        expect(client).to receive(:post).with("api/generate", hash_including(stream: true)).and_call_original
        response
      end

      it "yields the intermediate responses to the block", vcr: {cassette_name: "Langchain_LLM_Ollama_complete_when_passing_a_block_returns_a_completion"} do
        response
        expect(streamed_responses.length).to eq 4
        expect(streamed_responses).to be_all { |resp| resp.is_a?(Langchain::LLM::OllamaResponse) }
        expect(streamed_responses.map(&:completion).join).to eq("Complicated.")
      end
    end
  end

  describe "#chat" do
    let(:messages) { [{role: "user", content: "Hey! How are you?"}] }
    let(:response) { subject.chat(messages: messages) }

    it "returns a chat completion", :vcr do
      expect(response).to be_a(Langchain::LLM::OllamaResponse)
      expect(response.chat_completion).to include("I'm just a language model")
    end

    it "does not use streamed responses", vcr: {cassette_name: "Langchain_LLM_Ollama_chat_returns_a_chat_completion"} do
      expect(client).to receive(:post).with("api/chat", hash_including(stream: false)).and_call_original
      response
    end

    context "when passing a block" do
      let(:response) { subject.chat(messages: messages) { |resp| streamed_responses << resp } }
      let(:streamed_responses) { [] }

      it "returns a chat completion", :vcr do
        expect(response).to be_a(Langchain::LLM::OllamaResponse)
        expect(response.chat_completion).to include("I'm just a language model")
      end

      it "uses streamed responses", vcr: {cassette_name: "Langchain_LLM_Ollama_chat_when_passing_a_block_returns_a_chat_completion"} do
        expect(client).to receive(:post).with("api/chat", hash_including(stream: true)).and_call_original
        response
      end

      it "yields the intermediate responses to the block", vcr: {cassette_name: "Langchain_LLM_Ollama_chat_when_passing_a_block_returns_a_chat_completion"} do
        response
        expect(streamed_responses.length).to eq 51
        expect(streamed_responses).to be_all { |resp| resp.is_a?(Langchain::LLM::OllamaResponse) }
        expect(streamed_responses.map(&:chat_completion).join).to include("I'm just a language model")
      end
    end
  end

  describe "#summarize" do
    let(:mary_had_a_little_lamb_text) {
      File.read("spec/fixtures/llm/ollama/mary_had_a_little_lamb.txt")
    }

    it "returns a summarization", :vcr do
      response = subject.summarize(text: mary_had_a_little_lamb_text)

      expect(response).to be_a(Langchain::LLM::OllamaResponse)
      expect(response.completion).not_to match(/summary/)
      expect(response.completion).to start_with("A young girl named Mary has a pet lamb")
    end
  end

  describe "#default_dimensions" do
    it "returns size of embeddings" do
      embeddings = described_class::EMBEDDING_SIZES
      embeddings.each_pair do |model, size|
        subject = described_class.new(url: default_url, default_options: {embedding_model: model})
        expect(subject.default_dimensions).to eq(size)
      end
    end
  end

  describe "#json_responses_chunk_handler" do
    let(:parsed_chunks) { [] }
    let(:block) { proc { |chunk| parsed_chunks << chunk } }
    let(:handler) { subject.send(:json_responses_chunk_handler, &block) }

    describe "normal JSON chunks" do
      it "handles complete JSON objects on single lines" do
        chunk_data = %Q[{"response": "Hello"}\n{"response": "World"}]

        handler.call(chunk_data, chunk_data.size)

        expect(parsed_chunks).to eq([
          {"response" => "Hello"},
          {"response" => "World"}
        ])
      end

      it "handles complete JSON arrays" do
        chunk_data = %Q[{"items": [1, 2, 3]}\n{"items": ["a", "b", "c"]}]

        handler.call(chunk_data, chunk_data.size)

        expect(parsed_chunks).to eq([
          {"items" => [1, 2, 3]},
          {"items" => ["a", "b", "c"]}
        ])
      end

      it "skips empty lines" do
        chunk_data = %Q[{"response": "Hello"}\n\n{"response": "World"}\n]

        handler.call(chunk_data, chunk_data.size)

        expect(parsed_chunks).to eq([
          {"response" => "Hello"},
          {"response" => "World"}
        ])
      end
    end

    describe "incomplete JSON chunks" do
      it "handles JSON objects split across 2 chunks" do
        # First chunk with incomplete JSON
        first_chunk = %Q[{"response": "Hel]
        handler.call(first_chunk, first_chunk.size)
        expect(parsed_chunks).to be_empty

        # Second chunk completes the JSON
        second_chunk = %Q[lo", "done": true}]
        handler.call(second_chunk, second_chunk.size)

        expect(parsed_chunks).to eq([{"response" => "Hello", "done" => true}])
      end

      it "handles JSON objects split across multiple chunks" do
        # First chunk
        first_chunk = %Q[{"response": "Hel]
        handler.call(first_chunk, first_chunk.size)
        expect(parsed_chunks).to be_empty

        # Second chunk
        second_chunk = %Q[lo", "meta]
        handler.call(second_chunk, second_chunk.size)
        expect(parsed_chunks).to be_empty

        # Third chunk completes the JSON
        third_chunk = %Q[data": {"tokens": 5}}]
        handler.call(third_chunk, third_chunk.size)

        expect(parsed_chunks).to eq([{"response" => "Hello", "metadata" => {"tokens" => 5}}])
      end



      it "handles nested objects split across chunks" do
        # First chunk with incomplete nested object
        first_chunk = %Q[{"user": {"name": "John", "profile": {]
        handler.call(first_chunk, first_chunk.size)
        expect(parsed_chunks).to be_empty

        # Second chunk completes the nested structure
        second_chunk = %Q["age": 30, "city": "SF"}}}]
        handler.call(second_chunk, second_chunk.size)

        expect(parsed_chunks).to eq([{"user" => {"name" => "John", "profile" => {"age" => 30, "city" => "SF"}}}])
      end

      it "handles mixed complete and incomplete chunks" do
        # First chunk has complete JSON + incomplete
        first_chunk = %Q[{"response": "Hello"}\n{"partial": "dat]
        handler.call(first_chunk, first_chunk.size)
        expect(parsed_chunks).to eq([{"response" => "Hello"}])

        # Second chunk completes the incomplete JSON
        second_chunk = %Q[a", "done": true}]
        handler.call(second_chunk, second_chunk.size)

        expect(parsed_chunks).to eq([
          {"response" => "Hello"},
          {"partial" => "data", "done" => true}
        ])
      end
    end

    describe "malformed JSON handling" do
      it "raises JSON::ParserError for truly malformed JSON" do
        chunk_data = %Q[{"response": "Hello" invalid}]

        expect {
          handler.call(chunk_data, chunk_data.size)
        }.to raise_error(JSON::ParserError)
      end

      it "raises error for JSON with proper ending but invalid syntax" do
        chunk_data = %Q[{"response": "Hello",, "done": true}]

        expect {
          handler.call(chunk_data, chunk_data.size)
        }.to raise_error(JSON::ParserError)
      end

      it "handles complex malformed JSON that looks complete" do
        chunk_data = %Q[{"response": "Hello", "metadata": { "tokens": }}]

        expect {
          handler.call(chunk_data, chunk_data.size)
        }.to raise_error(JSON::ParserError)
      end
    end

    describe "memory safety" do
      it "raises error when incomplete buffer exceeds maximum size" do
        # Create a large incomplete JSON chunk (2MB + 100 bytes)
        large_chunk = '{"response": "' + 'x' * (2 * 1024 * 1024 + 100)

        expect {
          handler.call(large_chunk, large_chunk.size)
          handler.call('"}', 2)
        }.to raise_error(/Incomplete JSON buffer exceeded maximum size/)
      end

      it "handles moderate sized incomplete chunks within limits" do
        # Create a reasonable sized incomplete chunk
        moderate_chunk = '{"response": "' + 'x' * 10000
        handler.call(moderate_chunk, moderate_chunk.size)
        expect(parsed_chunks).to be_empty

        # Complete it
        handler.call('"}', 2)
        expect(parsed_chunks.size).to eq(1)
        expect(parsed_chunks.first["response"]).to start_with('x' * 10000)
      end
    end

    describe "logging behavior" do
      let(:logger) { double("logger") }

      before do
        allow(Langchain).to receive(:logger).and_return(logger)
        allow(logger).to receive(:debug)
        allow(logger).to receive(:error)
      end

      it "logs debug message for incomplete chunks" do
        chunk_data = %Q[{"response": "incomplete]

        handler.call(chunk_data, chunk_data.size)

        expect(logger).to have_received(:debug).with(/JSON chunk appears incomplete/)
      end

      it "logs error message for malformed JSON" do
        chunk_data = %Q[{"response": "Hello" invalid}]

        expect {
          handler.call(chunk_data, chunk_data.size)
        }.to raise_error(JSON::ParserError)

        expect(logger).to have_received(:error).with(/JSON parse error for chunk/)
      end

      it "logs error for buffer size exceeded" do
        large_chunk = '{"response": "' + 'x' * (2 * 1024 * 1024 + 100)

        expect {
          handler.call(large_chunk, large_chunk.size)
          handler.call('"}', 2)
        }.to raise_error(/Incomplete JSON buffer exceeded maximum size/)

        expect(logger).to have_received(:error).with(/Incomplete JSON buffer exceeded maximum size/)
      end
    end
  end

  describe "#looks_incomplete?" do
    it "identifies incomplete objects ending with comma" do
      expect(subject.send(:looks_incomplete?, '{"key": "value",')).to be true
    end

    it "identifies incomplete objects ending with colon" do
      expect(subject.send(:looks_incomplete?, '{"key":')).to be true
    end

    it "identifies incomplete arrays ending with comma" do
      expect(subject.send(:looks_incomplete?, '[1, 2,')).to be true
    end

    it "identifies incomplete objects with opening brace" do
      expect(subject.send(:looks_incomplete?, '{"key": {')).to be true
    end

    it "identifies incomplete arrays with opening bracket" do
      expect(subject.send(:looks_incomplete?, '{"items": [')).to be true
    end

    it "identifies objects with unmatched braces" do
      expect(subject.send(:looks_incomplete?, '{"outer": {"inner": "value"')).to be true
    end

    it "identifies arrays with unmatched brackets" do
      expect(subject.send(:looks_incomplete?, '[{"key": "value"')).to be true
    end

    it "recognizes complete objects" do
      expect(subject.send(:looks_incomplete?, '{"key": "value"}')).to be false
    end

    it "recognizes complete arrays" do
      expect(subject.send(:looks_incomplete?, '[1, 2, 3]')).to be false
    end

    it "recognizes complete nested structures" do
      expect(subject.send(:looks_incomplete?, '{"outer": {"inner": [1, 2, 3]}}')).to be false
    end

    it "handles whitespace properly" do
      expect(subject.send(:looks_incomplete?, '  {"key": "value"}  ')).to be false
      expect(subject.send(:looks_incomplete?, '  {"key": "value",  ')).to be true
    end
  end
end
