# frozen_string_literal: true

require_relative 'lib/litsea/version'

Gem::Specification.new do |spec|
  spec.name = 'litsea'
  spec.version = Litsea::VERSION
  spec.authors = ['Minoru Osuka']
  spec.summary = 'Ruby binding for Litsea, a compact word segmentation and POS tagging library'
  spec.description = 'Ruby binding for Litsea: word segmentation and Universal POS tagging ' \
                     'for Japanese, Chinese, Korean, and English. Models are not bundled.'
  spec.homepage = 'https://github.com/mosuka/litsea'
  spec.license = 'MIT'
  spec.required_ruby_version = '>= 3.1'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = spec.homepage
  spec.metadata['bug_tracker_uri'] = "#{spec.homepage}/issues"
  spec.metadata['rubygems_mfa_required'] = 'true'

  # A source gem: `gem install` compiles the crate with extconf.rb. Build it
  # with `bundle exec rake build`, which runs from the crate as `cargo package`
  # normalizes it and adds the repository's LICENSE. Running `gem build` in
  # this directory would ship its Cargo.toml, which only resolves inside the
  # Cargo workspace, and no LICENSE.
  spec.files = Dir[
    'lib/**/*.rb',
    'extconf.rb',
    'src/**/*.rs',
    'Cargo.toml',
    'LICENSE',
    'README.md',
    'README_ja.md'
  ]
  spec.extensions = ['extconf.rb']
  spec.require_paths = ['lib']

  spec.add_dependency 'rb_sys', '~> 0.9'
end
